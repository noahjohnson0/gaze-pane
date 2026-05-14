"""Voice control: continuous mic -> Silero VAD -> MLX Whisper -> wake/end parse.

Audio is captured by `sounddevice` (PortAudio/CoreAudio) into 32 ms frames.
A Silero VAD iterator emits 'start' and 'end' events; between them we buffer
audio. On 'end', the buffered utterance is handed to a transcription worker
that runs MLX Whisper on Apple-Silicon GPU and then looks for the user's
wake phrase, command, and end phrase inside the transcript.

Matched commands go into `command_queue` for the iTerm2 asyncio loop to send
into the active pane via `Session.async_send_text(cmd + '\\n')`.
"""
from __future__ import annotations

import queue
import re
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd


def _ensure_mic_permission(timeout_s: float = 30.0) -> bool:
    """Trigger macOS Microphone TCC prompt via AVCaptureDevice and wait for it.

    sounddevice/PortAudio uses CoreAudio HAL which on macOS doesn't reliably
    surface the TCC prompt for non-bundle binaries -- the mic returns silence
    without ever asking. AVCaptureDevice.requestAccess is the right API to
    trigger the prompt (cv2 already uses it for the camera path, which is why
    camera prompts work).
    """
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
    except ImportError as e:
        print(f"[voice] AVFoundation unavailable: {e!r}; mic permission "
              "prompt may not fire", file=sys.stderr)
        return True

    # AVAuthorizationStatus: 0=notDetermined, 1=restricted, 2=denied, 3=authorized
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)
    if status == 3:
        print(f"[{time.strftime('%H:%M:%S')}] [voice] mic permission ok "
              "(already granted, no prompt needed)", flush=True)
        return True
    if status == 2:
        print("[voice] Microphone permission denied. Grant it at:\n"
              "        System Settings -> Privacy & Security -> Microphone",
              file=sys.stderr)
        return False
    if status == 1:
        print("[voice] Microphone is restricted by system policy.",
              file=sys.stderr)
        return False

    print(f"[{time.strftime('%H:%M:%S')}] [voice] requesting microphone "
          "permission... (click Allow on the system prompt)", flush=True)
    event = threading.Event()
    granted: list[bool] = [False]

    def cb(allowed):
        granted[0] = bool(allowed)
        event.set()

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeAudio, cb)
    if not event.wait(timeout=timeout_s):
        print("[voice] mic permission prompt timed out", file=sys.stderr)
        return False
    if not granted[0]:
        print("[voice] microphone permission denied", file=sys.stderr)
        return False
    return True


SAMPLE_RATE = 16_000      # MLX Whisper expects 16 kHz mono float32
FRAME = 512               # Silero VAD requires 512-sample (32 ms) chunks at 16 kHz


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _strip_punct(s: str) -> str:
    # Drop characters that wouldn't make sense in a shell command (quotes can
    # be added later if we add an LLM-rewrite step), keep slashes/dots/dashes
    # for paths/flags.
    s = re.sub(r"[^\w\s./'\-]", "", s).strip()
    # Trailing/leading periods/commas/etc almost always come from the
    # surrounding sentence ("watermelon." or "ls -la."), strip them off the
    # ends without touching interior punctuation (paths, filenames).
    return s.strip(".,!?;: \t")


class VoiceListener:
    """Background voice listener. Start one, drain `command_queue` in your loop."""

    def __init__(
        self,
        *,
        wake_phrase: str = "hey claude",
        end_phrase: str = "send it",
        model: str = "mlx-community/whisper-small-mlx",
        device: Optional[int | str] = None,
        verbose: bool = True,
    ):
        self.wake = wake_phrase.lower().strip()
        self.end = end_phrase.lower().strip()
        self.model = model
        self.device = device
        self.verbose = verbose

        self.command_queue: queue.Queue[str] = queue.Queue()
        self._utt_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        self._mic_stream: Optional[sd.InputStream] = None
        self._workers: list[threading.Thread] = []

        # Lazy: set up in start().
        self._mlx = None
        self._vad = None
        self._vad_state = "idle"
        self._utt_buffer: list[np.ndarray] = []

        # Wake-phrase state machine. We accumulate across utterances so the
        # user can pause between wake / command / end.
        self._cmd_state = "idle"        # "idle" or "recording"
        self._cmd_buffer: str = ""
        self._cmd_started_at: float = 0.0
        self.cmd_timeout_s: float = 10.0

        # Diagnostic stats; printed every ~2s by a watcher thread.
        self._stats_lock = threading.Lock()
        self._stats_frames = 0
        self._stats_max_amp = 0.0
        self._stats_vad_starts = 0
        self._stats_vad_ends = 0

    # ---- public ----

    def start(self) -> None:
        # Trigger the macOS mic permission prompt before anything else so the
        # user sees it up front, not buried after a long whisper model load.
        if not _ensure_mic_permission():
            raise RuntimeError("microphone permission not granted")

        try:
            import mlx_whisper as mxw
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as e:
            print(f"[voice] missing dep: {e}\n"
                  f"  install: pip install mlx-whisper silero-vad sounddevice",
                  file=sys.stderr)
            raise

        self._mlx = mxw
        self._vad = VADIterator(load_silero_vad(), sampling_rate=SAMPLE_RATE)

        # Pre-warm Whisper. Without this, the first download/model-load (~500 MB
        # on first run, plus weights into memory on every run) happens during
        # the first utterance, which means the user speaks and nothing happens
        # for many seconds. A 1s silence buffer triggers the same code path.
        if self.verbose:
            print(f"[{_ts()}] [voice] loading {self.model} "
                  "(first run downloads ~500 MB from huggingface)...", flush=True)
        warm_t0 = time.time()
        try:
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            self._mlx.transcribe(silence, path_or_hf_repo=self.model, language="en")
        except Exception as e:
            print(f"[voice] model preload failed: {e!r}", file=sys.stderr)
            raise
        if self.verbose:
            print(f"[{_ts()}] [voice] model ready in {time.time()-warm_t0:.1f}s",
                  flush=True)

        worker = threading.Thread(
            target=self._transcribe_worker, name="voice-stt", daemon=True)
        worker.start()
        self._workers.append(worker)

        stats = threading.Thread(
            target=self._stats_worker, name="voice-stats", daemon=True)
        stats.start()
        self._workers.append(stats)

        # Show available input devices once so a wrong default is obvious.
        if self.verbose:
            print(f"[{_ts()}] [voice] audio input devices:", flush=True)
            try:
                default_in = sd.default.device[0] if isinstance(
                    sd.default.device, (list, tuple)) else sd.default.device
            except Exception:
                default_in = None
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    mark = " (default)" if i == default_in else ""
                    used = " (using)" if i == self.device else ""
                    print(f"        [{i}] {d['name']}  "
                          f"{d['max_input_channels']}ch{mark}{used}", flush=True)

        self._mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME, callback=self._audio_cb, device=self.device,
        )
        self._mic_stream.start()
        if self.verbose:
            print(f"[{_ts()}] [voice] listening: wake={self.wake!r}  end={self.end!r}",
                  flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=1.0)

    def get_command(self) -> Optional[str]:
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    # ---- mic callback (audio thread; do almost nothing here) ----

    def _audio_cb(self, indata: np.ndarray, frames: int, t, status) -> None:
        if status and self.verbose:
            # Underruns and similar — informational, not fatal.
            print(f"[voice] mic status: {status}", file=sys.stderr)
        frame = indata[:, 0].astype(np.float32).copy()
        amp = float(np.abs(frame).max())
        with self._stats_lock:
            self._stats_frames += 1
            if amp > self._stats_max_amp:
                self._stats_max_amp = amp
        evt = self._vad(frame, return_seconds=False)
        if evt is not None and "start" in evt:
            self._vad_state = "speaking"
            self._utt_buffer = [frame]
            with self._stats_lock:
                self._stats_vad_starts += 1
        elif self._vad_state == "speaking":
            self._utt_buffer.append(frame)
        if evt is not None and "end" in evt and self._vad_state == "speaking":
            self._vad_state = "idle"
            audio = np.concatenate(self._utt_buffer)
            self._utt_buffer = []
            with self._stats_lock:
                self._stats_vad_ends += 1
            # Drop ultra-short blips (probably mouse clicks / coughs).
            if len(audio) >= SAMPLE_RATE * 0.3:
                self._utt_queue.put(audio)

    def _stats_worker(self) -> None:
        """Print mic / VAD activity every 3 s so we can diagnose silent streams."""
        while not self._stop.is_set():
            time.sleep(3.0)
            with self._stats_lock:
                frames = self._stats_frames
                amp = self._stats_max_amp
                starts = self._stats_vad_starts
                ends = self._stats_vad_ends
                self._stats_frames = 0
                self._stats_max_amp = 0.0
                self._stats_vad_starts = 0
                self._stats_vad_ends = 0
            if not self.verbose:
                continue
            print(f"[{_ts()}] [voice] mic: {frames} frames in 3s  "
                  f"max_amp={amp:.3f}  vad: {starts} starts, {ends} ends  "
                  f"state={self._vad_state}", flush=True)

    # ---- transcription worker (background thread) ----

    def _transcribe_worker(self) -> None:
        while not self._stop.is_set():
            try:
                audio = self._utt_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self._mlx.transcribe(
                    audio,
                    path_or_hf_repo=self.model,
                    language="en",
                )
                text = (result.get("text") or "").strip().lower()
            except Exception as e:
                print(f"[voice] transcribe error: {e!r}", file=sys.stderr)
                continue
            if self.verbose:
                print(f"[{_ts()}] [voice] heard: {text!r}", flush=True)
            self._process_transcript(text)

    # ---- wake/command/end state machine ----

    def _process_transcript(self, text: str) -> None:
        """Update wake/command/end state. Emits commands when end is matched."""
        if not text:
            return
        now = time.time()

        # Time-out stale recording so a missed end phrase doesn't leave us
        # buffering forever.
        if (self._cmd_state == "recording"
                and now - self._cmd_started_at > self.cmd_timeout_s):
            if self.verbose:
                print(f"[{_ts()}] [voice] command timeout, dropping buffer "
                      f"{self._cmd_buffer!r}", flush=True)
            self._cmd_state = "idle"
            self._cmd_buffer = ""

        if self._cmd_state == "idle":
            wake_idx = text.find(self.wake)
            if wake_idx < 0:
                return
            after_wake = text[wake_idx + len(self.wake):]
            end_idx = after_wake.find(self.end)
            if end_idx >= 0:
                # Whole thing in one utterance.
                self._dispatch(after_wake[:end_idx])
                return
            # Wake heard, no end yet -> start recording.
            self._cmd_state = "recording"
            self._cmd_buffer = after_wake.strip()
            self._cmd_started_at = now
            if self.verbose:
                print(f"[{_ts()}] [voice] wake heard, recording... "
                      f"buffer={self._cmd_buffer!r}", flush=True)
            return

        # state == recording
        end_idx = text.find(self.end)
        if end_idx >= 0:
            before_end = text[:end_idx]
            buf = (self._cmd_buffer + " " + before_end).strip()
            self._dispatch(buf)
            self._cmd_state = "idle"
            self._cmd_buffer = ""
            return
        # Just more command text -> append.
        self._cmd_buffer = (self._cmd_buffer + " " + text).strip()
        self._cmd_started_at = now
        if self.verbose:
            print(f"[{_ts()}] [voice] recording... buffer={self._cmd_buffer!r}",
                  flush=True)

    def _dispatch(self, raw: str) -> None:
        command = _strip_punct(raw).strip()
        if not command:
            return
        if self.verbose:
            print(f"[{_ts()}] [voice] -> command: {command!r}", flush=True)
        self.command_queue.put(command)
