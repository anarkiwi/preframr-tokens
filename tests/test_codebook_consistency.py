"""Consistency gate closing the codebook drift bug class: the mask, the validator, and the decoder
must reason about codebook liveness from one source (the CodebookFamily registry), the dead-ref action
is pinned to a single DEAD_REF_POLICY, every spec'd op resolves to a complete family, and each family's
decode-time writes stay within its declared macro contract (the trace_contract PoC, promoted).
"""

import numpy as np
import pytest

from preframr_tokens.macros import codebook
from preframr_tokens.macros.codebook import DEAD_REF_POLICY, family_for_op
from preframr_tokens.macros.decode import expand_ops
from preframr_tokens.macros.loops import expand_loops
from preframr_tokens.macros.macro_contracts import CONTRACTS, Effect, reg_class
from preframr_tokens.macros.op_contracts import CODEBOOK_SPECS, OP_PRODUCER
from preframr_tokens.macros.state import DecodeState, _build_decode_state
from preframr_tokens.macros.validators import validate_codebook_refs
from preframr_tokens.macros.walker import FrameWalker
from preframr_tokens.stfconstants import STAMP_REF_OP
from tests.test_codebook_machine_equivalence import _CORPUS, _Builder

_FAMILY_CORPUS = {
    "stamp": "stamp_abs",
    "wavetable": "wt_def_ref",
    "instrument": "instrument_def_ref",
}


def test_registry_completeness():
    """Every op in CODEBOOK_SPECS resolves to a registered family with a complete codec -- the guard
    that an upstream-added codebook op cannot ship without a CodebookFamily + codec."""
    for op in CODEBOOK_SPECS:
        fam = family_for_op(op)
        assert (
            fam is not None
        ), f"op {op} in CODEBOOK_SPECS has no CodebookFamily — register one in codebook.py"
        codec = codebook._CODECS[fam.name]
        for method in ("def_open", "step", "commit", "ref"):
            assert callable(getattr(codec, method, None)), (op, method)


def test_dead_ref_policy_is_drop():
    """The dead-ref action is a single named policy; decode honours it (silent drop)."""
    assert DEAD_REF_POLICY == "drop"


def test_dead_ref_decode_drops_silently():
    """A REF to a never-defined id decodes without error and writes nothing on the target reg."""
    df = _Builder().frame().write(0, 99, op=STAMP_REF_OP).frame().df()
    out = expand_ops(df.copy())
    assert 0 not in out["reg"].tolist()


def test_dead_ref_validator_raises():
    """The offline validator is strict on the same stream the decoder drops -- the documented
    asymmetry (lenient decode, strict validation) the three facets now agree on via the registry.
    """
    df = _Builder().frame().write(0, 99, op=STAMP_REF_OP).frame().df()
    with pytest.raises(AssertionError):
        validate_codebook_refs(df)


class _RecArray(np.ndarray):
    """ndarray that reports each scalar __setitem__ index to its owning tracing state."""

    def __new__(cls, base, owner):
        obj = np.asarray(base).view(cls)
        obj._owner = owner
        return obj

    def __array_finalize__(self, obj):
        self._owner = getattr(obj, "_owner", None)

    def __setitem__(self, idx, val):
        owner = getattr(self, "_owner", None)
        if owner is not None and isinstance(idx, (int, np.integer)):
            owner.record_write(int(idx))
        np.ndarray.__setitem__(self, idx, val)


class _TracingState(DecodeState):
    """DecodeState that attributes every last_val write to the op currently in effect."""

    def install_trace(self):
        self.active_op = None
        self.writes_by_op = {}
        self.last_val = _RecArray(self.last_val, self)

    def record_write(self, reg):
        self.writes_by_op.setdefault(self.active_op, set()).add(reg)

    def tick_frame(self):
        prev, self.active_op = self.active_op, "REPLAY"
        try:
            return super().tick_frame()
        finally:
            self.active_op = prev


class _TraceWalker(FrameWalker):
    emit_synthetic_frame_marker = True

    def before_row(self, i, reg, op):
        self.state.active_op = int(op)
        return True


def _observed_writes(encoded_df, family_ops):
    """Decode ``encoded_df`` tracing last_val mutations, return the {(RegClass, Effect.REPLAY)} the
    family's ops (and the deferred REPLAY tick they schedule) introduce."""
    df = expand_loops(encoded_df.copy())
    base = _build_decode_state(df)
    state = _TracingState(base.frame_diff, last_diff=base.last_diff, strict=base.strict)
    state.install_trace()
    _TraceWalker(df, state).walk()
    regs = set()
    for op, written in state.writes_by_op.items():
        if op in family_ops or op == "REPLAY":
            regs |= written
    out = set()
    for reg in regs:
        rc = reg_class(reg)
        if rc is not None:
            out.add((rc, Effect.REPLAY))
    return out


@pytest.mark.parametrize("name", sorted(_FAMILY_CORPUS))
def test_observed_writes_subset_of_contract(name):
    """Each family's decode-time writes (traced through DecodeState) stay within its declared macro
    contract -- the decoder never introduces a write-class the contract does not list.
    """
    fam = codebook.CODEBOOK_FAMILIES[name]
    producer = OP_PRODUCER[fam.def_op]
    declared = CONTRACTS[producer].writes
    observed = _observed_writes(_CORPUS[_FAMILY_CORPUS[name]], fam.ops)
    assert observed, f"{name}: traced no writes (corpus stream did not exercise it)"
    assert observed <= declared, (
        f"{name}: decoder writes {sorted((rc.name, e.name) for rc, e in observed)} "
        f"exceed contract {sorted((rc.name, e.name) for rc, e in declared)}"
    )
