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
    """Pick the hand backend whose playroutine matches, or raise.

    GoatTracker is the only remaining hand backend (the gt2reloc single-speed
    shape, play = init + 3); it is kept for reference. Anything it does not match
    has no hand backend -- use the generic path (single-file ``.sid`` recovery via
    ``preframr_tokens.bacc.generic.recover_from_sid``), which is driver-agnostic.
    """
    from preframr_tokens.bacc.backends.goattracker import GoatTrackerBackend

    backend = GoatTrackerBackend()
    if backend.matches(psid):
        return backend
    raise ValueError(
        "no hand backend matches PSID "
        f"(load={psid.load_addr:#06x} init={psid.init_addr:#06x} "
        f"play={psid.play_addr:#06x}); use the generic path"
    )
