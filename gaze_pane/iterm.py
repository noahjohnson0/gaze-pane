"""iTerm2 Python API helpers: enumerate pane bounds and activate a pane.

iTerm2's Session has no `async_get_frame`. Instead, we reconstruct each pane's
rect from the splitter tree (`tab.root`) using each session's `grid_size` in
cells as proportional weights. Window chrome (title bar + tab bar) is deducted
from the window's full Quartz frame so the splitter math applies to the actual
pane content area.

Coordinates we expose are top-left normalized to the main screen (0..1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import iterm2
from iterm2 import Session, Splitter

# Default chrome: title bar + tab bar at top of a stock iTerm2 window, in points.
# Users with different settings (tab bar off, status bar at bottom, etc) can
# override via the run-time flag.
DEFAULT_CHROME_TOP = 52.0
DEFAULT_CHROME_BOTTOM = 0.0
DEFAULT_CHROME_LEFT = 0.0
DEFAULT_CHROME_RIGHT = 0.0


@dataclass
class PaneRect:
    session_id: str
    norm_left: float
    norm_top: float
    norm_right: float
    norm_bottom: float
    session: "iterm2.Session"

    def contains(self, nx: float, ny: float) -> bool:
        return (self.norm_left <= nx <= self.norm_right
                and self.norm_top <= ny <= self.norm_bottom)

    @property
    def area(self) -> float:
        return max(0.0, self.norm_right - self.norm_left) * \
               max(0.0, self.norm_bottom - self.norm_top)


def _measure(node: Union[Session, Splitter]) -> tuple[float, float]:
    """Cells (w, h) occupied by this subtree. Used as proportional weights."""
    if isinstance(node, Session):
        gs = node.grid_size
        return (max(float(gs.width), 1.0), max(float(gs.height), 1.0))
    # Splitter
    child_m = [_measure(c) for c in node.children]
    if not child_m:
        return (1.0, 1.0)
    if node.vertical:
        # vertical divider: children sit side by side
        return (sum(c[0] for c in child_m), max(c[1] for c in child_m))
    # horizontal divider: children stacked vertically
    return (max(c[0] for c in child_m), sum(c[1] for c in child_m))


def _layout(
    node: Union[Session, Splitter],
    x: float, y: float, w: float, h: float,
) -> dict[str, tuple[float, float, float, float]]:
    """Assign (x, y, w, h) in normalized [0..1] within the content area."""
    if isinstance(node, Session):
        return {node.session_id: (x, y, w, h)}
    if not node.children:
        return {}
    child_m = [_measure(c) for c in node.children]
    out: dict[str, tuple[float, float, float, float]] = {}
    if node.vertical:
        total = sum(m[0] for m in child_m)
        if total <= 0:
            return {}
        x0 = x
        for child, m in zip(node.children, child_m):
            cw = w * (m[0] / total)
            out.update(_layout(child, x0, y, cw, h))
            x0 += cw
    else:
        total = sum(m[1] for m in child_m)
        if total <= 0:
            return {}
        y0 = y
        for child, m in zip(node.children, child_m):
            ch = h * (m[1] / total)
            out.update(_layout(child, x, y0, w, ch))
            y0 += ch
    return out


async def get_pane_rects(
    connection: "iterm2.Connection",
    screen_w: float,
    screen_h: float,
    *,
    chrome_top: float = DEFAULT_CHROME_TOP,
    chrome_bottom: float = DEFAULT_CHROME_BOTTOM,
    chrome_left: float = DEFAULT_CHROME_LEFT,
    chrome_right: float = DEFAULT_CHROME_RIGHT,
) -> tuple[list[PaneRect], Optional[str]]:
    """Return (rects for current tab's panes, active session id).

    Rects are in top-left normalized screen coords (0..1).
    """
    app = await iterm2.async_get_app(connection)
    win = app.current_terminal_window
    if win is None:
        return [], None
    tab = win.current_tab
    if tab is None:
        return [], None
    root = tab.root
    if root is None:
        return [], None

    win_frame = await win.async_get_frame()
    wx = win_frame.origin.x
    wy = win_frame.origin.y           # Quartz: distance from bottom of screen
    ww = win_frame.size.width
    wh = win_frame.size.height

    # Window rect in top-left-origin screen coords.
    win_left = wx
    win_top = screen_h - (wy + wh)

    # Content area (window minus chrome).
    content_left = win_left + chrome_left
    content_top = win_top + chrome_top
    content_w = max(1.0, ww - chrome_left - chrome_right)
    content_h = max(1.0, wh - chrome_top - chrome_bottom)

    norm_layout = _layout(root, 0.0, 0.0, 1.0, 1.0)
    sessions_by_id = {s.session_id: s for s in root.sessions}

    rects: list[PaneRect] = []
    for sid, (nx, ny, nw, nh) in norm_layout.items():
        sess = sessions_by_id.get(sid)
        if sess is None:
            continue
        left = (content_left + nx * content_w) / screen_w
        top = (content_top + ny * content_h) / screen_h
        right = (content_left + (nx + nw) * content_w) / screen_w
        bottom = (content_top + (ny + nh) * content_h) / screen_h
        rects.append(PaneRect(
            session_id=sid,
            norm_left=left, norm_top=top,
            norm_right=right, norm_bottom=bottom,
            session=sess,
        ))

    active_id = tab.current_session.session_id if tab.current_session else None
    return rects, active_id


async def activate_session(session: "iterm2.Session") -> None:
    """Focus a session without yanking the window forward or switching tabs."""
    await session.async_activate(select_tab=False, order_window_front=False)
