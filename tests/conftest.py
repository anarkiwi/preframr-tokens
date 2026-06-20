"""Shared fixtures: resolve + recover the Monty_on_the_Run BACC program once."""

import os
import subprocess
import urllib.request

import pytest

from preframr_tokens import CPF, per_frame_state, recover_program

_LOCAL_SID = (
    "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid"
)
_LOCAL_DUMP = "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.1.dump.parquet"
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_fixtures")
_CACHE_SID = os.path.join(_FIXTURE_DIR, "Monty_on_the_Run.sid")
_CACHE_DUMP = os.path.join(_FIXTURE_DIR, "Monty_on_the_Run.1.dump.parquet")
_SID_URL = os.environ.get(
    "MONTY_SID_URL",
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid",
)


def _resolve():
    sid = next((p for p in (_LOCAL_SID, _CACHE_SID) if os.path.exists(p)), None)
    if sid is None:
        os.makedirs(_FIXTURE_DIR, exist_ok=True)
        urllib.request.urlretrieve(_SID_URL, _CACHE_SID)
        sid = _CACHE_SID
    dump = next((p for p in (_LOCAL_DUMP, _CACHE_DUMP) if os.path.exists(p)), None)
    if dump is None:
        os.makedirs(_FIXTURE_DIR, exist_ok=True)
        rc = subprocess.call(
            ["sidtrace", "--sid", sid, "--subtune", "1", "--out", _CACHE_DUMP]
        )
        assert rc == 0 and os.path.exists(_CACHE_DUMP), "sidtrace render failed"
        dump = _CACHE_DUMP
    return sid, dump


@pytest.fixture(scope="session")
def monty_paths():
    return _resolve()


@pytest.fixture(scope="session")
def monty_state(monty_paths):
    return per_frame_state(monty_paths[1], CPF, 10**9)


@pytest.fixture(scope="session")
def monty_program(monty_paths):
    return recover_program(monty_paths[0], monty_paths[1], CPF)


_TT_LOCAL_SID = (
    "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/5_Title_Tunes.sid"
)
_TT_LOCAL_DUMP = "/scratch/preframr/hvsc/C64Music/MUSICIANS/H/Hubbard_Rob/5_Title_Tunes.2.dump.parquet"
_TT_CACHE_SID = os.path.join(_FIXTURE_DIR, "5_Title_Tunes.sid")
_TT_CACHE_DUMP = os.path.join(_FIXTURE_DIR, "5_Title_Tunes.2.dump.parquet")
_TT_SID_URL = os.environ.get(
    "TT_SID_URL",
    "https://hvsc.brona.dk/HVSC/C64Music/MUSICIANS/H/Hubbard_Rob/5_Title_Tunes.sid",
)


def _resolve_tt():
    sid = next((p for p in (_TT_LOCAL_SID, _TT_CACHE_SID) if os.path.exists(p)), None)
    if sid is None:
        os.makedirs(_FIXTURE_DIR, exist_ok=True)
        urllib.request.urlretrieve(_TT_SID_URL, _TT_CACHE_SID)
        sid = _TT_CACHE_SID
    dump = next(
        (p for p in (_TT_LOCAL_DUMP, _TT_CACHE_DUMP) if os.path.exists(p)), None
    )
    if dump is None:
        os.makedirs(_FIXTURE_DIR, exist_ok=True)
        rc = subprocess.call(
            ["sidtrace", "--sid", sid, "--subtune", "2", "--out", _TT_CACHE_DUMP]
        )
        assert rc == 0 and os.path.exists(_TT_CACHE_DUMP), "sidtrace render failed"
        dump = _TT_CACHE_DUMP
    return sid, dump


@pytest.fixture(scope="session")
def tt_paths():
    return _resolve_tt()


@pytest.fixture(scope="session")
def tt_state(tt_paths):
    return per_frame_state(tt_paths[1], CPF, 10**9)


@pytest.fixture(scope="session")
def tt_program(tt_paths):
    return recover_program(tt_paths[0], tt_paths[1], CPF, subtune=1)
