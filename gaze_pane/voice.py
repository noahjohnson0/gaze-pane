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


SAMPLE_RATE = 16_000      # MLX Whisper expects 16 kHz mono float32
FRAME = 512               # Silero VAD requires 512-sample (32 ms) chunks at 16 kHz


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s./'\-]", "", s).strip()


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

    # ---- public ----

    def start(self) -> None:
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
        if self.verbose:
            print(f"[{_ts()}] [voice] mlx-whisper ({self.model}) + silero-vad ready",
                  flush=True)

        worker = threading.Thread(
            target=self._transcribe_worker, name="voice-stt", daemon=True)
        worker.start()
        self._workers.append(worker)

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
        evt = self._vad(frame, return_seconds=False)
        if evt is not None and "start" in evt:
            self._vad_state = "speaking"
            self._utt_buffer = [frame]
        elif self._vad_state == "speaking":
            self._utt_buffer.append(frame)
        if evt is not None and "end" in evt and self._vad_state == "speaking":
            self._vad_state = "idle"
            audio = np.concatenate(self._utt_buffer)
            self._utt_buffer = []
            # Drop ultra-short blips (probably mouse clicks / coughs).
            if len(audio) >= SAMPLE_RATE * 0.3:
                self._utt_queue.put(audio)

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
            cmd = self._parse_command(text)
            if cmd is not None:
                if self.verbose:
                    print(f"[{_ts()}] [voice] -> command: {cmd!r}", flush=True)
                self.command_queue.put(cmd)

    # ---- wake/command/end parsing ----

    def _parse_command(self, text: str) -> Optional[str]:
        """Find `wake ... command ... end`. Return command or None."""
        if not text:
            return None
        # Allow the wake / end words to span any whitespace and lose punctuation.
        wake_idx = text.find(self.wake)
        if wake_idx < 0:
            return None
        after_wake = text[wake_idx + len(self.wake):]
        end_idx = after_wake.find(self.end)
        if end_idx < 0:
            return None
        command = after_wake[:end_idx]
        command = _strip_punct(command)
        return command or None
