"""Coarse per-section role tracker for layer-3 voice lanes (AGENT_TASK_melody_skeleton.md §4B,
role_lane_factorization.md): assign each voice a causal role (bass / mid / lead) per block by sustained
pitch rank, so the byte-exact reorder can emit accompaniment before melody (melody-last). The lead hops
~2.5x/tune, so assignment is coarse (per block, not per note). Control-aware (a silent/un-gated voice is
excluded), NEVER waveform-routed (the Facemorph guardrail)."""

from __future__ import annotations

import numpy as np

from preframr_tokens.macros.generator_fit import note_of
from preframr_tokens.stfconstants import GEN_FREQ_REGS, INSTR_OFF_CTRL, VOICES

__all__ = ["voice_note_series", "block_roles", "roles_for", "lead_changes"]

_GATE_BIT = 0x01


def voice_note_series(state, ref=0.0):
    """Per voice, the per-frame LUT note when the voice is sounding (gate on AND freq>8), else None.
    Reads only freq + the control-register gate bit -- never the waveform nibble."""
    out = {}
    n = int(state.shape[0])
    for v, b in enumerate(GEN_FREQ_REGS):
        freq = (
            state[:, b].astype("int64") + 256 * state[:, b + 1].astype("int64")
        ).tolist()
        gate = (state[:, b + INSTR_OFF_CTRL].astype("int64") & _GATE_BIT).tolist()
        series = []
        for i in range(n):
            f = int(freq[i])
            series.append(note_of(f, ref) if (gate[i] and f > 8) else None)
        out[v] = series
    return out


def _rank_roles(median_notes):
    """Map ``{voice: median_note}`` (sounding voices only) to ``{voice: role}`` by pitch: lowest=bass,
    highest=lead, the rest mid. A lone sounding voice is the lead; ties break by voice index.
    """
    voices = sorted(median_notes, key=lambda v: (median_notes[v], v))
    roles = {}
    for rank, v in enumerate(voices):
        if rank == 0 and len(voices) > 1:
            roles[v] = "bass"
        elif rank == len(voices) - 1:
            roles[v] = "lead"
        else:
            roles[v] = "mid"
    return roles


def block_roles(state, block_frames=256, ref=0.0):
    """Per block of ``block_frames`` frames, ``{voice: role}`` from each voice's median sounding note in
    that block (a voice silent across the block is omitted). Coarse by design -- one assignment per block,
    the harmonic window the reorder emits bass..lead within."""
    series = voice_note_series(state, ref)
    n = int(state.shape[0])
    out = []
    for lo in range(0, max(1, n), block_frames):
        hi = min(n, lo + block_frames)
        med = {}
        for v in range(VOICES):
            sounding = [x for x in series[v][lo:hi] if x is not None]
            if sounding:
                med[v] = float(np.median(sounding))
        out.append(_rank_roles(med) if med else {})
    return out


def roles_for(state, ref=0.0):
    """One ``{voice: role}`` over the whole block (the voice-lane reorder operates per block): each
    sounding voice's median note ranked bass..lead. Silent voices omitted; never waveform-routed.
    """
    series = voice_note_series(state, ref)
    med = {}
    for v in range(VOICES):
        sounding = [x for x in series[v] if x is not None]
        if sounding:
            med[v] = float(np.median(sounding))
    return _rank_roles(med)


def lead_changes(roles_per_block):
    """Number of times the lead voice changes across blocks -- the work order's ~2.5x/tune evidence that
    role assignment must be coarse (per block) yet can hop, so fixed physical voice-lanes are wrong.
    """
    leads = []
    for roles in roles_per_block:
        lead = next((v for v, r in roles.items() if r == "lead"), None)
        if lead is not None:
            leads.append(lead)
    return sum(1 for a, b in zip(leads, leads[1:]) if a != b)
