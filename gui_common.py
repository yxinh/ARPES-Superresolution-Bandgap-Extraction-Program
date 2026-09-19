"""Shared Tkinter helpers for the three analysis tabs."""

from __future__ import annotations

import sys


def mousewheel_delta(event):
    """Convert a mouse-wheel event to Tk ``yview_scroll`` units."""
    if event.num == 4:
        return -1
    if event.num == 5:
        return 1
    if event.delta == 0:
        return 0
    if sys.platform == "darwin":
        return -1 if event.delta > 0 else 1
    return int(-1 * (event.delta / 120))


def bind_mousewheel(canvas, handler):
    """Scroll ``canvas`` only while the pointer is over it."""

    def _bind(_event=None):
        canvas.bind_all("<MouseWheel>", handler)
        canvas.bind_all("<Button-4>", handler)
        canvas.bind_all("<Button-5>", handler)

    def _unbind(_event=None):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _bind)
    canvas.bind("<Leave>", _unbind)
    return _bind, _unbind
