"""Driver backend interface + dispatch.

A backend encapsulates one playroutine's driver-specific knowledge: how to
recover a BaccProgram from the running emulator (white-box taps) and how to
render that program back to byte-exact per-frame registers.
"""

from abc import ABC, abstractmethod


class DriverBackend(ABC):
    name = "abstract"

    @abstractmethod
    def matches(self, psid):
        """True iff this backend handles the given PSID's playroutine."""

    @abstractmethod
    def recover(self, psid, nframes, subtune):
        """Run the playroutine, tap its state, return a BaccProgram."""

    @abstractmethod
    def render(self, program):
        """Render a BaccProgram to an (nframes, 25) int register array."""


def select_backend(psid):
    """Pick the backend whose playroutine matches, or raise (no silent fallback)."""
    from preframr_tokens.bacc.backends.hubbard import (
        Hubbard5TTBackend,
        HubbardMontyBackend,
    )

    for backend in (HubbardMontyBackend(), Hubbard5TTBackend()):
        if backend.matches(psid):
            return backend
    raise ValueError(
        f"no BACC driver backend matches PSID (load={psid.load_addr:#06x} "
        f"init={psid.init_addr:#06x} play={psid.play_addr:#06x})"
    )
