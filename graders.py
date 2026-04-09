"""Compatibility shim for legacy imports.

The canonical graders live in env.graders. Keep this module as a thin
re-export so older imports still work without duplicating scoring logic.
"""

from env.graders import *  # noqa: F401,F403
