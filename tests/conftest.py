"""Shared fixtures: resolve + recover the BACC programs once per session.

Fixtures acquire (.sid, .dump) via tests._dump_fixture (download the .sid, render
the dump in the anarkiwi/headlessvice container, cache both -- no committed dumps).
"""

import pytest

from tests._dump_fixture import acquire

_GALWAY_REL = "MUSICIANS/G/Galway_Martin/Times_of_Lore.sid"
_GALWAY_URL = (
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/G/Galway_Martin/Times_of_Lore.sid"
)


@pytest.fixture(scope="session")
def galway_paths():
    """A clearly-multispeed Galway tune (~3x play-calls per raster frame)."""
    return acquire(_GALWAY_REL, _GALWAY_URL, subtune=1)
