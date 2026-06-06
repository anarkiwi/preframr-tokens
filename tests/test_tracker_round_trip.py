"""Forward tracker round-trip: module to register log to generator tokenizer to decode equals
player output. Renders real SID-Wizard / defMON modules through their own players, parses the
log under the deployed default with ``parse_audit='raise'`` (the sanctioned byte-exact oracle),
so completing the parse is the same-output guarantee. Skips when the optional tracker deps or
fixtures are absent."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

from tests.tracker_render import render_to_parquet

NFRAMES = int(os.environ.get("TRACKER_RT_NFRAMES", "1200"))
MAX_FIXTURES = int(os.environ.get("TRACKER_RT_MAX_FIXTURES", "6"))


def _swm_fixtures():
    env = os.environ.get("PREFRAMR_SWM_DIR")
    cands = (
        sorted(glob.glob(os.path.join(env, "*.swm")))
        if env and os.path.isdir(env)
        else []
    )
    return cands[:MAX_FIXTURES]


def _defmon_fixtures():
    env = os.environ.get("PREFRAMR_DEFMON_DIR")
    if env and os.path.isdir(env):
        return sorted(glob.glob(os.path.join(env, "*.prg")))[:MAX_FIXTURES]
    try:
        import pydefmon

        bundled = Path(pydefmon.__file__).resolve().parent.parent / "build" / "fixtures"
        return sorted(glob.glob(str(bundled / "*.prg")))[:MAX_FIXTURES]
    except Exception:
        return []


def _assert_round_trip(kind, src_path, tmp_path):
    """Render then parse under full_macros with parse_audit raise; reaching the end without an
    AssertionError is the byte-exact / same-output guarantee."""
    pytest.importorskip("preframr_tokens")
    from preframr_tokens.reglogparser import RegLogParser
    from preframr_tokens.tokenizer_config import named_config

    name = Path(src_path).stem
    try:
        dump = render_to_parquet(
            kind, src_path, Path(tmp_path) / f"{name}.dump.parquet", NFRAMES
        )
    except Exception as exc:
        if type(exc).__name__ in {"DefmonError", "SWMError", "SWMFormatError"}:
            pytest.skip(f"{kind} {name}: player cannot load this fixture ({exc})")
        raise

    parser = RegLogParser(args=named_config("full_macros", parse_audit="raise"))
    blocks = 0
    try:
        for _ in parser.parse(str(dump), max_perm=1, require_pq=False, reparse=True):
            blocks += 1
    except AssertionError as exc:
        pytest.fail(
            f"{kind} {name}: generator round-trip NOT lossless -- {str(exc)[:300]}"
        )
    assert blocks >= 1, f"{kind} {name}: parser yielded no token blocks"


@pytest.mark.parametrize(
    "src",
    _swm_fixtures()
    or [
        pytest.param(
            None, marks=pytest.mark.skip(reason="no SWM fixtures; set PREFRAMR_SWM_DIR")
        )
    ],
)
def test_swm_forward_round_trip(src, tmp_path):
    pytest.importorskip("pysidwizard")
    _assert_round_trip("swm", src, tmp_path)


@pytest.mark.parametrize(
    "src",
    _defmon_fixtures()
    or [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="no defMON fixtures; set PREFRAMR_DEFMON_DIR"
            ),
        )
    ],
)
def test_defmon_forward_round_trip(src, tmp_path):
    pytest.importorskip("pydefmon")
    _assert_round_trip("defmon", src, tmp_path)


@pytest.mark.xfail(
    reason="log->SWM recompiler unbuilt; the full SWM->log->SWM->log loop is the reverse half",
    strict=False,
)
def test_reverse_recompile_full_loop():
    """Placeholder for the reverse half (render to log to recompiled SWM to log, identical).
    Requires the register-log to SWM recompiler, which is designed but not implemented.
    """
    pytest.importorskip("pysidwizard")
    raise NotImplementedError("register-log -> SWM recompiler not implemented")
