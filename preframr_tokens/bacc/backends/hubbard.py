"""Rob Hubbard "Monty-class" playroutine backend (load $8000, play $8012).

Recovers a sparse note-on score + instrument table + initial-state seed by
tapping driver RAM under py65, and renders it byte-exact by reimplementing the
per-frame generator pass (vibrato/pulse/portamento/drums/skydive/octave-arp +
the gate/release rule). Validated residual-zero on Monty_on_the_Run.
"""

import numpy as np

from preframr_tokens.bacc.backends.base import DriverBackend
from preframr_tokens.bacc.primitive import BaccProgram, NoteOn
from preframr_tokens.bacc.sidemu import SIDEmu, NREG

# Driver RAM map (Monty-class player; working vars at $84C0+, note table $8400).
NOTETAB = 0x8400
INSTR = 0x93B4
N_INSTR = 20
A_LEN, A_LNTH, A_CTRL, A_NOTE, A_INSTR = 0x84CA, 0x84CD, 0x84D0, 0x84D3, 0x84D6
A_PDLY, A_PDIR, A_SFH, A_SFL, A_PORTA = 0x84E5, 0x84E8, 0x84EF, 0x84F2, 0x84F5
A_SPEED, A_RESETSPD = 0x84EB, 0x84EC
# address -> tracked per-voice seed key, for note indices that read past the table
_RAM_VARS = (
    (A_LEN, "lenleft"),
    (A_LNTH, "lnthcc"),
    (A_CTRL, "vctrl"),
    (A_NOTE, "notenum"),
    (A_INSTR, "instrnr"),
    (A_PDLY, "pdly"),
    (A_PDIR, "pdir"),
    (A_SFH, "sfh"),
    (A_SFL, "sfl"),
    (A_PORTA, "porta"),
)
_SEED_KEYS = (
    "notenum",
    "instrnr",
    "lnthcc",
    "lenleft",
    "sfl",
    "sfh",
    "porta",
    "vctrl",
    "pdly",
    "pdir",
)


class HubbardMontyBackend(DriverBackend):
    name = "hubbard_monty"

    def matches(self, psid):
        return psid.load_addr == 0x8000 and psid.play_addr == 0x8012

    def recover(self, psid, nframes, subtune):
        emu = SIDEmu(psid)
        emu.init(subtune)
        static_img = [emu.mem[NOTETAB + i] for i in range(256)]
        instruments = [
            [emu.mem[INSTR + i * 8 + j] for j in range(8)] for i in range(N_INSTR)
        ]
        seed = {
            k: [emu.mem[a + v] for v in range(3)]
            for a, k in (
                (A_NOTE, "notenum"),
                (A_INSTR, "instrnr"),
                (A_LNTH, "lnthcc"),
                (A_LEN, "lenleft"),
                (A_SFL, "sfl"),
                (A_SFH, "sfh"),
                (A_PORTA, "porta"),
                (A_CTRL, "vctrl"),
                (A_PDLY, "pdly"),
                (A_PDIR, "pdir"),
            )
        }
        seed["init_speed"] = emu.mem[A_SPEED]
        seed["resetspd"] = emu.mem[A_RESETSPD]
        emu.mem.watch = {A_LNTH + v for v in range(3)}
        score = []
        for k in range(nframes - 1):  # play#k produces frame k+1
            emu.play_frame()
            hit = {a for a, _ in emu.mem.watch_hits}
            for v in range(3):
                if (A_LNTH + v) in hit:
                    score.append(
                        NoteOn(
                            k,
                            v,
                            emu.mem[A_NOTE + v],
                            emu.mem[A_INSTR + v],
                            emu.mem[A_LNTH + v],
                            emu.mem[A_PORTA + v],
                        )
                    )
        return BaccProgram(
            self.name, nframes, [], instruments, score, seed, {"static_img": static_img}
        )

    def render(self, p):
        st = {k: list(p.seed[k]) for k in _SEED_KEYS}
        itab = [row[:] for row in p.instruments]
        static_img = p.tables["static_img"]
        resetspd = p.seed["resetspd"]
        by_frame = [[] for _ in range(p.nframes)]
        for ev in p.score:
            by_frame[ev.frame].append(ev)

        def live_byte(addr):
            if addr < A_LEN or addr >= 0x8500:
                return static_img[addr - NOTETAB] if NOTETAB <= addr < 0x8500 else 0
            for base, key in _RAM_VARS:
                if base <= addr < base + 3:
                    return st[key][addr - base] & 0xFF
            return static_img[addr - NOTETAB]

        def nfreq(note):
            a = NOTETAB + (note & 0xFF) * 2
            return live_byte(a) | (live_byte(a + 1) << 8)

        out = np.zeros((p.nframes, NREG), dtype=np.int64)
        out[0] = p.boot
        reg = list(p.boot)
        speed = p.seed["init_speed"]
        for k in range(p.nframes - 1):
            speed = resetspd if speed - 1 < 0 else speed - 1
            check = speed == resetspd
            evs = {ev.voice: ev for ev in by_frame[k]}
            for x in (2, 1, 0):
                y = 7 * x
                if x in evs:
                    _note_on(reg, y, x, evs[x], st, itab, nfreq)
                    continue
                if check:
                    st["lenleft"][x] -= 1
                    if not (st["lnthcc"][x] & 0x20) and st["lenleft"][x] == 0:
                        reg[y + 4] = st["vctrl"][x] & 0xFE
                        reg[y + 5] = 0
                        reg[y + 6] = 0
                _generators(reg, y, x, k & 0xFF, st, itab, nfreq)
            out[k + 1] = reg
        return out


def _note_on(reg, y, x, ev, st, itab, nfreq):
    st["lnthcc"][x] = ev.lnth
    st["porta"][x] = ev.porta
    st["instrnr"][x] = ev.instr
    ins = ev.instr
    if not ev.lnth & 0x40:  # not an append (legato) note: set pitch
        st["notenum"][x] = ev.note
        fr = nfreq(ev.note)
        st["sfl"][x] = fr & 0xFF
        st["sfh"][x] = (fr >> 8) & 0xFF
        reg[y + 0] = st["sfl"][x]
        reg[y + 1] = st["sfh"][x]
    ctrl = itab[ins][2]
    reg[y + 4] = ctrl & (0xFE if ev.lnth & 0x40 else 0xFF)
    reg[y + 2] = itab[ins][0]
    reg[y + 3] = itab[ins][1]
    reg[y + 5] = itab[ins][3]
    reg[y + 6] = itab[ins][4]
    st["vctrl"][x] = ctrl
    st["lenleft"][x] = ev.lnth & 0x1F


def _generators(
    reg, y, x, ctr, st, itab, nfreq, pulse_additive=False, skydive_gated=False
):
    ins = st["instrnr"][x]
    depth, pulsevalue, fx = itab[ins][5], itab[ins][6], itab[ins][7]
    carry = 0
    if depth != 0:  # vibrato: triangle phase x note-relative amplitude
        osc = ctr & 7
        if osc >= 4:
            osc ^= 7
        base = nfreq(st["notenum"][x])
        amp = (nfreq(st["notenum"][x] + 1) - base) & 0xFFFF
        for _ in range(depth + 1):
            amp >>= 1
        frq = base
        if (st["lnthcc"][x] & 0x1F) >= 8:
            # byte-wise 16-bit add (osc times); track carry-out for the no-CLC
            # simple-pw ADC that inherits it (freq -> pw coupling)
            if osc == 0:
                carry = 1
            lo, hi = base & 0xFF, (base >> 8) & 0xFF
            dlo, dhi = amp & 0xFF, (amp >> 8) & 0xFF
            for _ in range(osc):
                t = lo + dlo
                lo = t & 0xFF
                t = hi + dhi + (t >> 8)
                hi = t & 0xFF
                carry = t >> 8
            frq = lo | (hi << 8)
        reg[y + 0] = frq & 0xFF
        reg[y + 1] = (frq >> 8) & 0xFF
    if pulse_additive and (fx & 0x08):  # additive simple-pw (carry-coupled), pwlo only
        itab[ins][0] = (itab[ins][0] + pulsevalue + carry) & 0xFF
        reg[y + 2] = itab[ins][0]
    elif pulsevalue != 0:  # pulse-width: bounce hi-nibble between 0x08 and 0x0e
        st["pdly"][x] -= 1
        if st["pdly"][x] < 0:
            st["pdly"][x] = pulsevalue & 0x1F
            pspeed = pulsevalue & 0xE0
            if st["pdir"][x] != 0:
                hi = itab[ins][0] - pspeed
                lo = (itab[ins][1] - (1 if hi < 0 else 0)) & 0x0F
                hi &= 0xFF
                if lo == 0x08:
                    st["pdir"][x] = (st["pdir"][x] - 1) & 0xFF
            else:
                hi = itab[ins][0] + pspeed
                lo = (itab[ins][1] + (1 if hi > 0xFF else 0)) & 0x0F
                hi &= 0xFF
                if lo == 0x0E:
                    st["pdir"][x] = (st["pdir"][x] + 1) & 0xFF
            itab[ins][0] = hi
            itab[ins][1] = lo
            reg[y + 2] = hi
            reg[y + 3] = lo
    if st["porta"][x] != 0:  # portamento: accumulate +/- onto savefreq
        tmp = st["porta"][x] & 0x7E
        cur = (st["sfh"][x] << 8) | st["sfl"][x]
        cur = (cur - tmp if st["porta"][x] & 1 else cur + tmp) & 0xFFFF
        st["sfl"][x] = cur & 0xFF
        st["sfh"][x] = (cur >> 8) & 0xFF
        reg[y + 0] = st["sfl"][x]
        reg[y + 1] = st["sfh"][x]
    if fx & 1:  # drums: write OLD savefreqhi, store decremented
        if st["sfh"][x] != 0 and st["lenleft"][x] != 0:
            if (st["lnthcc"][x] & 0x1F) - 1 < st["lenleft"][x]:
                reg[y + 1] = st["sfh"][x]
                reg[y + 4] = 0x80
            else:
                old = st["sfh"][x]
                st["sfh"][x] = (old - 1) & 0xFF
                cw = st["vctrl"][x] & 0xFE
                if cw != 0:
                    reg[y + 1] = old
                    reg[y + 4] = cw
                else:
                    reg[y + 1] = st["sfh"][x]
                    reg[y + 4] = 0x80
    if fx & 2:  # skydive: write OLD savefreqhi, store decremented (every 2nd frame)
        ok = (ctr & 1) and st["sfh"][x] != 0
        if skydive_gated:  # later-version gate: only near the note's tail
            ok = ok and (st["lnthcc"][x] & 0x1F) >= 0x10 and st["lenleft"][x] < 0x18
        if ok:
            old = st["sfh"][x]
            st["sfh"][x] = (old - 1) & 0xFF
            reg[y + 1] = old
    if fx & 4:  # octave arpeggio: note vs note+12 on the frame counter
        note = st["notenum"][x] + (0x0C if (ctr & 1) else 0)
        reg[y + 0] = nfreq(note) & 0xFF
        reg[y + 1] = (nfreq(note) >> 8) & 0xFF


# --- 5_Title_Tunes subtune 2: same player, block-2 RAM map (Monty - 0x67f9) ---
_TT_DELTA = -0x67F9
TT_NOTETAB, TT_INSTR = 0x1C07, 0x1D02
TT_LEN, TT_LNTH, TT_CTRL = A_LEN + _TT_DELTA, A_LNTH + _TT_DELTA, A_CTRL + _TT_DELTA
TT_NOTE, TT_INSTRNR = A_NOTE + _TT_DELTA, A_INSTR + _TT_DELTA
TT_PDLY, TT_PDIR = A_PDLY + _TT_DELTA, A_PDIR + _TT_DELTA
TT_SFH, TT_SFL, TT_PORTA = A_SFH + _TT_DELTA, A_SFL + _TT_DELTA, A_PORTA + _TT_DELTA
TT_SPEED, TT_RESETSPD = A_SPEED + _TT_DELTA, A_RESETSPD + _TT_DELTA
TT_POS, TT_PAT = 0x84C4 + _TT_DELTA, 0x84C7 + _TT_DELTA  # posoffset, patoffset
_TT_RAM_VARS = (
    (TT_LEN, "lenleft"),
    (TT_LNTH, "lnthcc"),
    (TT_CTRL, "vctrl"),
    (TT_NOTE, "notenum"),
    (TT_INSTRNR, "instrnr"),
    (TT_PDLY, "pdly"),
    (TT_PDIR, "pdir"),
    (TT_SFH, "sfh"),
    (TT_SFL, "sfl"),
    (TT_PORTA, "porta"),
)


class Hubbard5TTBackend(DriverBackend):
    """5_Title_Tunes subtune-2 player (load $0b10): same Hubbard generators as
    Monty plus an additive carry-coupled simple-pw mode and a tail-gated skydive."""

    name = "hubbard_5tt"

    def matches(self, psid):
        return psid.load_addr == 0x0B10 and psid.play_addr == 0x0B40

    def recover(self, psid, nframes, subtune):
        emu = SIDEmu(psid)
        emu.init(subtune)
        static_img = [emu.mem[TT_NOTETAB + i] for i in range(256)]
        instruments = [
            [emu.mem[TT_INSTR + i * 8 + j] for j in range(8)] for i in range(8)
        ]
        emu.mem.watch = {TT_LNTH + v for v in range(3)}
        score, pos, pat = [], [], []
        seed, itab0, speed0 = None, None, None
        for k in range(nframes - 1):
            emu.play_frame()
            pos.append([emu.mem[TT_POS + i] for i in range(3)])
            pat.append([emu.mem[TT_PAT + i] for i in range(3)])
            hit = {a for a, _ in emu.mem.watch_hits}
            for v in range(3):
                if (TT_LNTH + v) in hit:
                    score.append(
                        NoteOn(
                            k,
                            v,
                            emu.mem[TT_NOTE + v],
                            emu.mem[TT_INSTRNR + v],
                            emu.mem[TT_LNTH + v],
                            emu.mem[TT_PORTA + v],
                        )
                    )
            if k == 0:  # boot frame 1 + generator state seed = post-play#0
                seed = {
                    key: [emu.mem[a + v] for v in range(3)] for a, key in _TT_RAM_VARS
                }
                itab0 = [
                    [emu.mem[TT_INSTR + i * 8 + j] for j in range(8)] for i in range(8)
                ]
                speed0 = emu.mem[TT_SPEED]
        seed["itab"] = itab0
        seed["init_speed"] = speed0
        seed["resetspd"] = emu.mem[TT_RESETSPD]
        return BaccProgram(
            self.name,
            nframes,
            [],
            instruments,
            score,
            seed,
            {"static_img": static_img, "pos": pos, "pat": pat},
        )

    def render(self, p):
        st = {key: list(p.seed[key]) for _, key in _TT_RAM_VARS}
        itab = [row[:] for row in p.seed["itab"]]
        pos, pat, static_img = p.tables["pos"], p.tables["pat"], p.tables["static_img"]
        resetspd = p.seed["resetspd"]
        by_frame = [[] for _ in range(p.nframes)]
        for ev in p.score:
            by_frame[ev.frame].append(ev)
        cur = [0]

        def live_byte(addr):
            if TT_POS <= addr < TT_POS + 3:
                return pos[cur[0]][addr - TT_POS]
            if TT_PAT <= addr < TT_PAT + 3:
                return pat[cur[0]][addr - TT_PAT]
            for base, key in _TT_RAM_VARS:
                if base <= addr < base + 3:
                    return st[key][addr - base] & 0xFF
            i = addr - TT_NOTETAB
            return static_img[i] if 0 <= i < 256 else 0

        def nfreq(note):
            a = TT_NOTETAB + (note & 0xFF) * 2
            return live_byte(a) | (live_byte(a + 1) << 8)

        out = np.zeros((p.nframes, NREG), dtype=np.int64)
        out[0] = p.boot
        out[1] = p.tables["boot1"]
        reg = list(p.tables["boot1"])
        speed = p.seed["init_speed"]
        for f in range(2, p.nframes):
            k = f - 1
            cur[0] = k
            speed = resetspd if speed - 1 < 0 else speed - 1
            check = speed == resetspd
            evs = {ev.voice: ev for ev in by_frame[k]}
            for x in (2, 1, 0):
                y = 7 * x
                if x in evs:
                    _note_on(reg, y, x, evs[x], st, itab, nfreq)
                    continue
                if check:
                    st["lenleft"][x] -= 1
                    if not (st["lnthcc"][x] & 0x20) and st["lenleft"][x] == 0:
                        reg[y + 4] = st["vctrl"][x] & 0xFE
                        reg[y + 5] = 0
                        reg[y + 6] = 0
                _generators(
                    reg,
                    y,
                    x,
                    k & 0xFF,
                    st,
                    itab,
                    nfreq,
                    pulse_additive=True,
                    skydive_gated=True,
                )
            out[f] = reg
        return out
