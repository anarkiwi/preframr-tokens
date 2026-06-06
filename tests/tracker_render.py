"""Render a tracker module (SID-Wizard SWM / defMON .prg) through its own bit-exact
player into a preframr register-write dump (the ``clock,irq,chipno,reg,val`` schema
``RegLogParser`` ingests). Front half of the forward round-trip test."""

from __future__ import annotations

D400 = 0xD400
MAX_REG = 24
PAL_FRAME_PERIOD = 19656


def _reg_off(reg: int) -> int:
    return reg - D400 if reg >= D400 else reg


def render_swm(path, nframes: int):
    """Per-tick list of (reg_offset, value) writes from the SID-Wizard player."""
    from pysidwizard import read_swm
    from pysidwizard.player import SWMPlayer

    return _drive(SWMPlayer(read_swm(str(path))), nframes)


def render_defmon(path, nframes: int):
    """Per-tick list of (reg_offset, value) writes from the defMON player."""
    from pydefmon import DefmonSong, DefmonPlayer

    return _drive(DefmonPlayer(DefmonSong.from_file(str(path))), nframes)


def _drive(player, nframes: int):
    frames = []
    for _ in range(nframes):
        writes = []
        for reg, val in player.play_frame():
            off = _reg_off(int(reg))
            if 0 <= off <= MAX_REG:
                writes.append((off, int(val) & 0xFF))
        frames.append(writes)
    return frames


def frames_to_dump_df(frames, period: int = PAL_FRAME_PERIOD):
    """Frame-major writes to a dump DataFrame in the parser ingest schema. Each tick with
    writes shares one ``irq=(tick+1)*period``; empty ticks become timing gaps (parser
    DELAYs); intra-tick write order is preserved."""
    import pandas as pd

    clocks, irqs, regs, vals = [], [], [], []
    for tick, writes in enumerate(frames):
        if not writes:
            continue
        irq = (tick + 1) * period
        for k, (reg, val) in enumerate(writes):
            clocks.append(irq + k + 1)
            irqs.append(irq)
            regs.append(reg)
            vals.append(val)
    return pd.DataFrame(
        {
            "clock": pd.array(clocks, dtype="UInt32"),
            "irq": pd.array(irqs, dtype="UInt32"),
            "chipno": pd.array([0] * len(clocks), dtype="UInt8"),
            "reg": pd.array(regs, dtype="UInt8"),
            "val": pd.array(vals, dtype="UInt8"),
        }
    )


def render_to_parquet(kind: str, src_path, out_path, nframes: int):
    """Render ``src_path`` (kind 'swm' | 'defmon') to a dump parquet at ``out_path``."""
    frames = (
        render_swm(src_path, nframes)
        if kind == "swm"
        else render_defmon(src_path, nframes)
    )
    df = frames_to_dump_df(frames)
    if len(df) == 0:
        raise RuntimeError(f"{src_path}: player produced no register writes")
    df.to_parquet(str(out_path))
    return out_path
