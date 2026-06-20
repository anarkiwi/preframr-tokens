"""Shared fixtures: resolve + recover the BACC programs once per session.

Fixtures acquire (.sid, .dump) via tests._dump_fixture (download the .sid, render
the dump in the anarkiwi/headlessvice container, cache both -- no committed dumps).
"""

import pytest

from preframr_tokens import CPF, per_frame_state, recover_program
from tests._dump_fixture import acquire

_MONTY_REL = "MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid"
_MONTY_URL = (
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid"
)
_TT_REL = "MUSICIANS/H/Hubbard_Rob/5_Title_Tunes.sid"
_TT_URL = (
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/H/Hubbard_Rob/5_Title_Tunes.sid"
)
_GALWAY_REL = "MUSICIANS/G/Galway_Martin/Times_of_Lore.sid"
_GALWAY_URL = (
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/G/Galway_Martin/Times_of_Lore.sid"
)


@pytest.fixture(scope="session")
def monty_paths():
    return acquire(_MONTY_REL, _MONTY_URL, subtune=1)


@pytest.fixture(scope="session")
def monty_state(monty_paths):
    return per_frame_state(monty_paths[1], CPF, 10**9)


@pytest.fixture(scope="session")
def monty_program(monty_paths):
    return recover_program(monty_paths[0], monty_paths[1], CPF)


@pytest.fixture(scope="session")
def tt_paths():
    return acquire(_TT_REL, _TT_URL, subtune=2)


@pytest.fixture(scope="session")
def tt_state(tt_paths):
    return per_frame_state(tt_paths[1], CPF, 10**9)


@pytest.fixture(scope="session")
def tt_program(tt_paths):
    return recover_program(tt_paths[0], tt_paths[1], CPF, subtune=1)


@pytest.fixture(scope="session")
def galway_paths():
    """A clearly-multispeed Galway tune (~3x play-calls per raster frame)."""
    return acquire(_GALWAY_REL, _GALWAY_URL, subtune=1)
