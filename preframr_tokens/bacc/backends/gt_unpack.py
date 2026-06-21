"""Byte-exact GoatTracker2-packed-SID decompiler.

Reconstructs a pygoattracker ``Song`` from the relocated/packed player image
embedded in a PSID, inverting greloc's *value* transforms (wavetable waveform
+$10 offset, notetbl ^$80 for normal-note rows, filter passband bits, SETTEMPO
databyte) while keeping every index/pointer in greloc's PACKED numbering. The
packed numbering is a self-consistent bijection on used entries, so feeding
pygoattracker the packed numbers everywhere renders identically to the original
.sng -- without inverting instrmap/pattmap/tablemap.

The layout is AUTO-DERIVED (no per-song hardcoding):

  * freq anchor: longest run where image == FREQ_LO[first:first+L] immediately
    followed by FREQ_HI[first:first+L] -> freqtbllo addr, firstnote, L.
  * songs = psid.songs.  songtbl (songs*3, lo||hi) and patttbl (P, lo||hi)
    follow contiguously; both hold ABSOLUTE addresses (orderlists / patterns).
  * P (patterns) = largest P whose patttbl entries all land in the pattern
    region (after the last orderlist).
  * I (instruments) = max instrument index used across patterns.
  * Instrument SoA columns + the eight table bases are read straight from the
    relocated player's indexed operands (each base B is loaded as ``B-1,y``;
    a table base is additionally loaded as ``B,y``).  ncols, the 4 table
    lengths and the speed leading-zero flag come out uniquely from those.
  * Ambiguous instrument flags (vibrato cols vs gate cols when exactly one
    extra pair is present) are resolved by reconstructing under each candidate
    and selecting the one whose render matches the real player (py65).

``render_state`` renders per-frame 25-register SID state via pygoattracker and
drops the leading INIT frame (pygoattracker folds song-init into play_frame #0,
which emits no register writes; the real player does init separately), so the
output aligns 1:1 with the real player's per-frame output.
"""

import numpy as np
from pygoattracker import constants as C
from pygoattracker.model import (
    Instrument,
    Orderlist,
    Pattern,
    Row,
    Song,
    Subtune,
    Table,
    entry_from_byte,
)
from pygoattracker.player import Player

from preframr_tokens.bacc.sidemu import load_psid, SIDEmu

SID_PATH = "/scratch/preframr/hvsc/C64Music/MUSICIANS/J/Jammer/Grid_Runner.sid"
DUMP_PATH = "/scratch/tmp/gtwork/Grid_Runner.1.dump.parquet"

NREG = 25
_PW_HI = (3, 10, 17)

FREQ_LO = C.FREQ_LO
FREQ_HI = C.FREQ_HI

# 6502 opcodes whose operand is a 16-bit absolute address (the player loads
# every table/SoA column through one of these as ``addr,x`` / ``addr,y``).
_IDX_OPS = {
    0xB9,
    0xBD,
    0xBE,
    0x79,
    0xD9,
    0x99,
    0x9D,
    0xDD,
    0xB6,
    0xBC,
    0xDE,
    0x59,
    0x19,
    0x39,
}


# =========================== image accessor ================================
class _Image:
    def __init__(self, path):
        self.psid = load_psid(path)
        self.load = self.psid.load_addr
        self.data = self.psid.data

    def __getitem__(self, addr):
        return self.data[addr - self.load]

    def slice(self, addr, n):
        off = addr - self.load
        return list(self.data[off : off + n])

    def end(self):
        return self.load + len(self.data)


# =========================== layout derivation =============================
def _freq_anchor(img):
    """(freqtbllo_addr, firstnote, L) via the longest FREQ_LO||FREQ_HI run."""
    data = img.data
    best = None
    NF = len(FREQ_LO)
    for first in range(NF):
        for L in range(NF - first, 0, -1):
            pat = bytes(FREQ_LO[first : first + L]) + bytes(FREQ_HI[first : first + L])
            idx = data.find(pat)
            if idx >= 0:
                if best is None or L > best[1]:
                    best = (idx, L, first)
                break
    off, L, first = best
    return img.load + off, first, L


def _parse_orderlist_end(img, base):
    a = base
    while img[a] != C.LOOPSONG:
        a += 1
    return a + 2  # past the LOOPSONG byte and the restart byte


def _operand_refs(img, lo, hi):
    """Map operand-address -> count for every absolute-indexed operand whose
    address falls in [lo-3, hi)."""
    data = img.data
    refs = {}
    for o in range(len(data) - 2):
        if data[o] in _IDX_OPS:
            a = data[o + 1] | (data[o + 2] << 8)
            if lo - 3 <= a < hi:
                refs[a] = refs.get(a, 0) + 1
    return refs


def _instr_count(img, patttbllo, P):
    """Largest instrument index referenced across all patterns (greloc emits
    exactly that many instruments, 1-based)."""
    lo = img.slice(patttbllo, P)
    hi = img.slice(patttbllo + P, P)
    addrs = [(h << 8) | l for l, h in zip(lo, hi)]
    maxi = 0
    for a in addrs:
        p = a
        while True:
            b = img[p]
            p += 1
            if b == 0x00:
                break
            if b < 0x40:  # instrument change
                maxi = max(maxi, b)
            elif 0x40 <= b < 0x50:  # FX + note
                if b & 0x0F:
                    p += 1
                p += 1
            elif 0x50 <= b < 0x60:  # FX only
                if b & 0x0F:
                    p += 1
            # 0x60-0xBF bare note, 0xC0+ packed rest: no operand bytes
    return maxi


def _derive_layout(img):
    """Return a dict with every absolute address / count / table length the
    reconstruction needs, derived purely from the image."""
    songs = img.psid.songs
    fl_addr, firstnote, L = _freq_anchor(img)
    lastnote = firstnote + L - 1

    songtbllo = fl_addr + 2 * L
    n_song = songs * 3
    songlo = img.slice(songtbllo, n_song)
    songhi = img.slice(songtbllo + n_song, n_song)
    orderlist_addrs = [(h << 8) | l for l, h in zip(songlo, songhi)]

    patttbllo = songtbllo + 2 * n_song
    order_start = min(orderlist_addrs)
    ol_end = max(_parse_orderlist_end(img, b) for b in orderlist_addrs)

    maxP = (order_start - patttbllo) // 2
    P = None
    for cand in range(maxP, 0, -1):
        lo = img.slice(patttbllo, cand)
        hi = img.slice(patttbllo + cand, cand)
        addrs = [(h << 8) | l for l, h in zip(lo, hi)]
        if all(ol_end <= x < img.end() for x in addrs):
            P = cand
            break

    instr_base = patttbllo + 2 * P
    I = _instr_count(img, patttbllo, P)

    refs = _operand_refs(img, instr_base, order_start)

    # wave_L = first instr_base + k*I that is loaded both as base and base-1
    # (only table bases are loaded with a 0 offset; SoA columns never are).
    wave_L = None
    ncols = 0
    k = 0
    while instr_base + k * I < order_start:
        W = instr_base + k * I
        if W in refs and (W - 1) in refs:
            wave_L = W
            ncols = k
            break
        k += 1
    if wave_L is None:
        raise ValueError("could not locate the wavetable base in the player image")
    cols = [instr_base + j * I for j in range(ncols)]

    # Split [wave_L, order_start) into the 8 contiguous tables (4 equal-length
    # L/R pairs, plus the speed leading-zero).  Each table base B satisfies
    # ``(B-1) in refs``; brute force the 4 lengths under that constraint.
    def is_base(b):
        return (b - 1) in refs

    split = None
    region = order_start - wave_L
    for wl in range(1, region):
        note_R = wave_L + wl
        pulse_L = note_R + wl
        if pulse_L >= order_start:
            break
        if not (is_base(note_R) and is_base(pulse_L)):
            continue
        for pl in range(1, region):
            pulse_R = pulse_L + pl
            filt_L = pulse_R + pl
            if filt_L >= order_start:
                break
            if not (is_base(pulse_R) and is_base(filt_L)):
                continue
            for fl in range(1, region):
                filt_R = filt_L + fl
                speed_area = filt_R + fl
                if speed_area >= order_start:
                    break
                if not is_base(filt_R):
                    continue
                rem = order_start - speed_area
                for zbytes in (0, 2):  # speed leading zero each side
                    if (rem - zbytes) % 2 or (rem - zbytes) < 2:
                        continue
                    sl = (rem - zbytes) // 2
                    z = 1 if zbytes else 0
                    speed_L = speed_area + z
                    speed_R = speed_L + sl + z
                    if is_base(speed_L) and is_base(speed_R):
                        if split is not None:
                            raise RuntimeError("ambiguous table split")
                        split = dict(
                            wl=wl,
                            pl=pl,
                            fl=fl,
                            sl=sl,
                            speedzero=bool(z),
                            wave_L=wave_L,
                            note_R=note_R,
                            pulse_L=pulse_L,
                            pulse_R=pulse_R,
                            filt_L=filt_L,
                            filt_R=filt_R,
                            speed_L=speed_L,
                            speed_R=speed_R,
                        )
    if split is None:
        raise RuntimeError("could not split tables")

    lay = dict(
        songs=songs,
        firstnote=firstnote,
        lastnote=lastnote,
        L=L,
        freqtbllo=fl_addr,
        songtbllo=songtbllo,
        patttbllo=patttbllo,
        P=P,
        I=I,
        instr_base=instr_base,
        ncols=ncols,
        cols=cols,
        orderlist_addrs=orderlist_addrs,
        refs=refs,
        order_start=order_start,
    )
    lay.update(split)
    return lay


# ----------------------- instrument flag resolution ------------------------
def _chnwave_reg(img, wave_L):
    """Address of mt_chnwave (the per-channel current-waveform ghostreg).  The
    wave executor does ``sta mt_chnwave,x (9D R)`` immediately followed by
    ``lda mt_wavetbl,y (B9 wave_L)`` (player.s 940-942).  Anchor on that
    ``B9 wave_L`` and read the preceding ``9D R`` store target."""
    data = img.data
    target = wave_L & 0xFF, (wave_L >> 8) & 0xFF
    for o in range(3, len(data) - 2):
        if data[o] == 0xB9 and data[o + 1] == target[0] and data[o + 2] == target[1]:
            if data[o - 3] == 0x9D:
                return data[o - 2] | (data[o - 1] << 8)
    return None


def _recover_hr_order(img):
    """Recover (FIRSTNOHRINSTR, FIRSTLEGATOINSTR) from the player's hard-restart
    ordering check.  greloc emits (player.s ~1293-1325):

        lda mt_chninstr,x          ; BD lo hi
        cmp #FIRSTNOHRINSTR        ; C9 imm
        bcs mt_skiphr / mt_nohr_legato   ; B0 rel
        lda #SRPARAM ; sta chnsr,x ; lda #ADPARAM ; sta chnad,x  (the HR write)

    Instruments numbered >= FIRSTNOHRINSTR skip the hard restart (modelled in
    pygoattracker by gateoff_timer bit $80).  If a legato range exists a second
    ``cmp #FIRSTLEGATOINSTR ; bcc`` precedes the gate set.  Returns
    (firstnohr, firstlegato); either may be None (no such range)."""
    data = img.data
    firstnohr = None
    for o in range(len(data) - 12):
        # BD lo hi ; C9 imm ; B0 rel ; A9 ad ; 9D .. .. ; A9 sr ; 9D .. ..
        if (
            data[o] == 0xBD
            and data[o + 3] == 0xC9
            and data[o + 5] == 0xB0
            and data[o + 7] == 0xA9
            and data[o + 9] == 0x9D
            and data[o + 12] == 0xA9
            and data[o + 14] == 0x9D
        ):
            firstnohr = data[o + 4]
            break
    # The legato range only exists when the FIRSTNOHR check branches to
    # mt_nohr_legato, which is uniquely ``cmp #imm ; bcc mt_skiphr ; bcs mt_rest``
    # (a cmp immediately followed by BCC then BCS -- player.s 1390-1393).
    firstlegato = None
    for o in range(len(data) - 4):
        if data[o] == 0xC9 and data[o + 2] == 0x90 and data[o + 4] == 0xB0:
            firstlegato = data[o + 1]
            break
    return firstnohr, firstlegato


def _recover_fixed_params(img, wave_L):
    """For FIXEDPARAMS=1 builds, recover the global GATETIMERPARAM and
    FIRSTWAVEPARAM the player bakes in as immediates.

    FIRSTWAVE: ``lda #FIRSTWAVEPARAM (A9 imm); sta mt_chnwave,x (9D lo hi)``
    where mt_chnwave is the wavetable executor's store target.

    GATETIMER: the gate-off check ``lda mt_chncounter,x (BD lo hi);
    cmp #GATETIMERPARAM (C9 imm); beq mt_getnewnote (F0 rel)`` -- distinguished
    from the gate-timer reload check (``cmp #$03``) by the long forward branch
    to the new-note fetch.  Returns (gatetimer, firstwave) or (None, None)."""
    data = img.data
    chnwave = _chnwave_reg(img, wave_L)
    firstwave = None
    if chnwave is not None:
        for o in range(len(data) - 4):
            if (
                data[o] == 0xA9
                and data[o + 2] == 0x9D
                and (data[o + 3] | (data[o + 4] << 8)) == chnwave
            ):
                firstwave = data[o + 1]
                break
    gatetimer = None
    best_off = -1
    for o in range(len(data) - 6):
        if data[o] == 0xBD and data[o + 3] == 0xC9 and data[o + 5] == 0xF0:
            off = data[o + 6] if data[o + 6] < 128 else data[o + 6] - 256
            if off > best_off:  # gate-off -> getnewnote is the far jump
                best_off = off
                gatetimer = data[o + 4]
    return gatetimer, firstwave


def _col_sta_target(img, col_base):
    """STA target of the first store after ``LDA col_base-1,y`` in the player."""
    data = img.data
    op = col_base - 1
    for o in range(len(data) - 2):
        if data[o] == 0xB9 and (data[o + 1] | (data[o + 2] << 8)) == op:
            for q in range(o + 3, min(o + 18, len(data) - 2)):
                if data[q] in (0x9D, 0x99, 0x8D):
                    return data[q + 1] | (data[q + 2] << 8)
                if data[q] == 0x85:
                    return data[q + 1]
            return None
    return None


def _instrument_flags(img, lay):
    """Resolve (nopulse, nofilter, noinsvib, fixedparams) from ncols and the
    last column's STA target.  Column emit order is fixed:
        ad, sr, wave, [pulse], [filt], [vibparam, vibdelay], [gatetimer, firstwave]
    pulse/filter presence: detected by whether the pulse/filter TABLE exists
    (it always does here) -- assume present, then ncols-5 gives the extra pairs.
    """
    ncols = lay["ncols"]
    nopulse = 0
    nofilter = 0
    base = 3 + (0 if nopulse else 1) + (0 if nofilter else 1)
    extra = ncols - base  # 0, 2 or 4 extra columns
    if extra == 0:
        return dict(nopulse=nopulse, nofilter=nofilter, noinsvib=1, fixedparams=1)
    if extra == 4:
        return dict(nopulse=nopulse, nofilter=nofilter, noinsvib=0, fixedparams=0)
    # extra == 2: vibrato pair OR gate pair.  The gate pair's last column is
    # firstwave, stored to mt_chnwave; the vib pair's last column is vibdelay.
    chnwave = _chnwave_reg(img, lay["wave_L"])
    last_tgt = _col_sta_target(img, lay["cols"][-1])
    if chnwave is not None and last_tgt == chnwave:
        return dict(nopulse=nopulse, nofilter=nofilter, noinsvib=1, fixedparams=0)
    return dict(nopulse=nopulse, nofilter=nofilter, noinsvib=0, fixedparams=1)


# =========================== table inverses ================================
def _invert_wave_left(packed):
    out = []
    for p in packed:
        if p <= 0x0F:
            out.append(p)
        elif p <= 0x1F:
            out.append((p + 0xD0) & 0xFF)  # silent: was &$f then +$10
        elif p <= 0xEF:
            out.append((p - 0x10) & 0xFF)  # waveform: was +$10
        else:
            out.append(p)  # command / jump
    return out


def _invert_note_right(left_orig, packed_right):
    out = []
    for lo, r in zip(left_orig, packed_right):
        out.append(r if 0xF0 <= lo <= 0xFF else r ^ 0x80)
    return out


def _invert_filter_left(packed):
    out = []
    for p in packed:
        if p != 0xFF and p > 0x80:
            out.append((((p & 0x38) << 1) | 0x80) & 0xFF)
        else:
            out.append(p)
    return out


def _nowavedelay(img, wave_L):
    """True iff the build set NOWAVEDELAY=1 (no wavetable delay rows).  greloc
    then SKIPS the +$10 waveform offset, and the player's wave executor lacks
    the ``sbc #$10`` undo.  Detect by anchoring on ``lda mt_wavetbl-1,y
    (B9 wave_L-1)`` and checking whether a ``sbc #$10 (E9 10)`` follows within
    the executor (NOWAVEDELAY==0) or not (NOWAVEDELAY==1)."""
    data = img.data
    op = (wave_L - 1) & 0xFFFF
    lo, hi = op & 0xFF, (op >> 8) & 0xFF
    for o in range(len(data) - 2):
        if data[o] == 0xB9 and data[o + 1] == lo and data[o + 2] == hi:
            window = data[o + 3 : o + 20]
            for q in range(len(window) - 1):
                if window[q] == 0xE9 and window[q + 1] == 0x10:
                    return False  # sbc #$10 present -> NOWAVEDELAY=0
            return True
    return True


def _build_tables(img, lay):
    nowavedelay = _nowavedelay(img, lay["wave_L"])
    if nowavedelay:
        wl = img.slice(lay["wave_L"], lay["wl"])  # no +$10 was applied
    else:
        wl = _invert_wave_left(img.slice(lay["wave_L"], lay["wl"]))
    wr = _invert_note_right(wl, img.slice(lay["note_R"], lay["wl"]))
    wave = Table(left=wl, right=wr)

    pulse = Table(
        left=img.slice(lay["pulse_L"], lay["pl"]),
        right=img.slice(lay["pulse_R"], lay["pl"]),
    )

    fl = _invert_filter_left(img.slice(lay["filt_L"], lay["fl"]))
    filt = Table(left=fl, right=img.slice(lay["filt_R"], lay["fl"]))

    speed = Table(
        left=img.slice(lay["speed_L"], lay["sl"]),
        right=img.slice(lay["speed_R"], lay["sl"]),
    )
    return wave, pulse, filt, speed


# =========================== instruments ===================================
def _build_instruments(img, lay, flags):
    I = lay["I"]
    cols = lay["cols"]
    nopulse, nofilter = flags["nopulse"], flags["nofilter"]
    noinsvib, fixedparams = flags["noinsvib"], flags["fixedparams"]
    firstnohr, firstlegato = _recover_hr_order(img)

    # column index assignment (emit order)
    ci = 0
    c_ad, ci = cols[ci], ci + 1
    c_sr, ci = cols[ci], ci + 1
    c_wave, ci = cols[ci], ci + 1
    c_pulse = None
    if not nopulse:
        c_pulse, ci = cols[ci], ci + 1
    c_filt = None
    if not nofilter:
        c_filt, ci = cols[ci], ci + 1
    c_vibp = c_vibd = None
    if not noinsvib:
        c_vibp, ci = cols[ci], ci + 1
        c_vibd, ci = cols[ci], ci + 1
    c_gate = c_fw = None
    fixed_gate = fixed_fw = 0
    if not fixedparams:
        c_gate, ci = cols[ci], ci + 1
        c_fw, ci = cols[ci], ci + 1
    else:
        # FIXEDPARAMS=1: gatetimer/firstwave are a single global value baked
        # into the player as immediates rather than per-instrument columns.
        g, fw = _recover_fixed_params(img, lay["wave_L"])
        fixed_gate = (g or 0) & 0x3F
        fixed_fw = fw or 0

    def col(base, i):
        return img[base + i] if base is not None else 0

    instrs = []
    for i in range(I):
        instr_num = i + 1  # player numbers instruments 1-based
        if fixedparams:
            gatetimer = fixed_gate
            firstwave = fixed_fw
        else:
            gatetimer = col(c_gate, i) & 0x3F
            firstwave = col(c_fw, i)
        # Hard-restart instrument-ordering optimization: instruments numbered
        # >= FIRSTNOHRINSTR skip the hard restart (gateoff_timer bit $80);
        # instruments >= FIRSTLEGATOINSTR are legato (also no gate-off, bit $40).
        if firstnohr is not None and instr_num >= firstnohr:
            gatetimer |= 0x80
        if firstlegato is not None and instr_num >= firstlegato:
            gatetimer |= 0x40
        vibparam = col(c_vibp, i)
        # greloc stores vibdelay-1 when the instrument has vibrato (else 0,0);
        # invert by adding 1 back when a vibrato param is present.
        vibdelay = (col(c_vibd, i) + 1) & 0xFF if vibparam else 0
        instrs.append(
            Instrument(
                attack_decay=col(c_ad, i),
                sustain_release=col(c_sr, i),
                wave_ptr=col(c_wave, i),
                pulse_ptr=col(c_pulse, i),
                filter_ptr=col(c_filt, i),
                vibrato_param=vibparam,
                vibrato_delay=vibdelay,
                gateoff_timer=gatetimer,
                first_wave=firstwave,
                name=f"i{i + 1:02d}",
            )
        )
    return instrs


# =========================== orderlists ====================================
def _build_orderlists(img, lay):
    songs = lay["songs"]
    addrs = lay["orderlist_addrs"]
    subtunes = []
    for s in range(songs):
        chans = []
        for ch in range(3):
            base = addrs[s * 3 + ch]
            a = base
            raw = []
            while img[a] != C.LOOPSONG:
                raw.append(img[a])
                a += 1
            restart = img[a + 1]
            # greloc swaps repeat sequences: a repeat count $D1-$DF
            # ("> REPEAT") is emitted AFTER its pattern, but the .sng / editor
            # order is repeat-then-pattern (greloc.c ~655-666).  Undo the swap.
            order = []
            i = 0
            while i < len(raw):
                b = raw[i]
                if (
                    i + 1 < len(raw)
                    and C.REPEAT < raw[i + 1] < C.TRANSDOWN
                    and raw[i] < C.REPEAT
                ):
                    order.append(raw[i + 1])  # repeat first
                    order.append(raw[i])  # then its pattern
                    i += 2
                else:
                    order.append(b)
                    i += 1
            entries = [entry_from_byte(b) for b in order]
            chans.append(Orderlist(entries=entries, restart=restart))
        subtunes.append(Subtune(channels=chans))
    return subtunes


# =========================== patterns ======================================
FX, FXONLY, FIRSTNOTE, REST, ENDMARK = 0x40, 0x50, 0x60, 0xBD, 0x00


def _invert_settempo(command, data):
    if command == C.CMD_SETTEMPO and (data & 0x7F) >= 2:
        return (data + 1) & 0xFF
    return data


def _unpack_pattern(img, addr):
    rows = []
    instr = command = data = 0
    a = addr
    while True:
        b = img[a]
        a += 1
        if b == ENDMARK:
            break
        if b < FX:  # instrument change
            instr = b
            continue
        if FXONLY <= b < FIRSTNOTE:  # FX on a REST row
            command = b & 0x0F
            data = img[a] if command else 0
            if command:
                a += 1
            rows.append(
                Row(
                    note=REST,
                    instrument=instr,
                    command=command,
                    data=_invert_settempo(command, data),
                )
            )
            instr = 0
            continue
        if FX <= b < FXONLY:  # FX followed by a note
            command = b & 0x0F
            data = img[a] if command else 0
            if command:
                a += 1
            note = img[a]
            a += 1
            rows.append(
                Row(
                    note=note,
                    instrument=instr,
                    command=command,
                    data=_invert_settempo(command, data),
                )
            )
            instr = 0
            continue
        if FIRSTNOTE <= b < 0xC0:  # bare note / REST / KEYOFF / KEYON
            rows.append(
                Row(
                    note=b,
                    instrument=instr,
                    command=command,
                    data=_invert_settempo(command, data),
                )
            )
            instr = 0
            continue
        # b >= 0xC0: packed rest of (256-b) rows
        for _ in range(256 - b):
            rows.append(
                Row(
                    note=REST,
                    instrument=0,
                    command=command,
                    data=_invert_settempo(command, data),
                )
            )
    return Pattern(rows=rows)


def _build_patterns(img, lay):
    lo = img.slice(lay["patttbllo"], lay["P"])
    hi = img.slice(lay["patttbllo"] + lay["P"], lay["P"])
    addrs = [(h << 8) | l for l, h in zip(lo, hi)]
    return [_unpack_pattern(img, a) for a in addrs]


# =========================== ADPARAM =======================================
def _hr_store_imm_tgt(d, p):
    """Parse ``A9 imm ; STA <tgt>`` at offset ``p``: returns ``(imm, tgt, len)``
    or ``None``.  STA forms: ``9D lo hi`` (abs,x), ``99 lo hi`` (abs,y),
    ``95 zp`` (zp,x), ``85 zp`` (zp)."""
    if d[p] != 0xA9:
        return None
    imm = d[p + 1]
    q = p + 2
    if d[q] in (0x9D, 0x99):  # sta abs,x / abs,y
        return imm, d[q + 1] | (d[q + 2] << 8), (q + 3) - p
    if d[q] in (0x95, 0x85):  # sta zp,x / zp
        return imm, d[q + 1], (q + 2) - p
    return None


def _hr_store_len(d, o):
    """Length of an ``A9 imm ; STA ; A9 imm ; STA`` hard-restart pair at offset
    ``o`` (player.s mt_normalnote 1303-1318), or ``None`` if none starts here.
    Returns ``(srimm, adimm, total_len)``.

    The two store targets are CONSECUTIVE registers; the chip/RAM layout always
    places AD below SR (SIDBASE+$05 < SIDBASE+$06; ``mt_chnad`` before
    ``mt_chnsr``; ghostad before ghostsr), so the LOWER-address store is AD and
    the HIGHER-address store is SR -- read the immediate by its store TARGET,
    not by position (greloc emits the two stores in either order across
    player versions)."""
    first = _hr_store_imm_tgt(d, o)
    if first is None:
        return None
    imm1, tgt1, len1 = first
    second = _hr_store_imm_tgt(d, o + len1)
    if second is None:
        return None
    imm2, tgt2, len2 = second
    if abs(tgt1 - tgt2) != 1:  # not a consecutive ad/sr register pair
        return None
    # lower address == AD (chnad/SIDBASE+5/ghostad); higher == SR.
    if tgt1 < tgt2:
        adimm, srimm = imm1, imm2
    else:
        adimm, srimm = imm2, imm1
    return srimm, adimm, len1 + len2


def _adparam_from_image(img):
    """Recover (ADPARAM<<8)|SRPARAM straight from the player's hard-restart
    store -- the ground truth the chip executes (player.s mt_normalnote, 1303
    ``lda #SRPARAM`` then 1313 ``lda #ADPARAM``).  Returns 0 if no HR store is
    found (a NUMHRINSTR==0 build does no hard restart).

    The HR store sits in ``mt_normalnote`` immediately after the gateoff guard:
    ``beq mt_rest`` (TONEPORTA check, F0) or ``bcs mt_skiphr``/``mt_nohr_legato``
    (FIRSTNOHR check, B0).  Anchoring on that preceding conditional branch
    excludes same-shaped ``A9 imm ; sta ; A9 imm ; sta`` pairs in the init
    routines (which zero adjacent RAM registers behind an ``ldx`` loop)."""
    d = img.data
    for o in range(2, len(d) - 10):
        if d[o - 2] not in (0xB0, 0xF0):  # bcs / beq guard precedes the HR store
            continue
        hit = _hr_store_len(d, o)
        if hit is not None:
            srimm, adimm, _ = hit
            return (adimm << 8) | srimm
    return 0


def _adparam_from_py65(sid_path, nframes=24):
    """First nonzero AD/SR pair the real player writes is the hard-restart
    ADPARAM/SRPARAM.  Returns (AD<<8)|SR or 0.

    Fragile fallback only: a tune whose first note's real ADSR is emitted
    before any hard-restart frame returns the NOTE's ADSR, not ADPARAM -- so
    the image-anchored recovery (``_adparam_from_image``) is preferred."""
    try:
        emu = SIDEmu(load_psid(sid_path))
        emu.init(0)
        for _ in range(nframes):
            emu.play_frame()
            s = emu.state()
            for v in range(3):
                ad, sr = s[5 + 7 * v], s[6 + 7 * v]
                if ad or sr:
                    return (ad << 8) | sr
    except Exception:
        pass
    return 0


def _find_adparam_static(img):
    """Legacy fallback: two adjacent hard-restart stores; defaults to the
    GoatTracker stock ``ad=$0F, sr=$00`` when no store is found."""
    return _adparam_from_image(img) or 0x0F00


# =========================== optimization flags ============================
def _gateoff_candidates(img):
    """Every ``lda mt_chncounter,x ; cmp <gatetimer> ; beq <rel>`` site.

    The player emits three such ``BD .. ; cmp ; F0 rel`` sequences: two gate-
    timer *reload* checks (``cmp #$03``) that bracket the gateoff-timer check
    (``cmp #GATETIMERPARAM`` when FIXEDPARAMS, else ``cmp mt_chngatetimer,x``).
    Returns the list of (offset, branch_rel) sorted by offset."""
    d = img.data
    cands = []
    for o in range(len(d) - 7):
        if d[o] == 0xBD and d[o + 3] == 0xC9 and d[o + 5] == 0xF0:  # cmp #imm
            rel = d[o + 6] - 256 if d[o + 6] >= 128 else d[o + 6]
            cands.append((o, rel))
        elif d[o] == 0xBD and d[o + 3] == 0xDD and d[o + 6] == 0xF0:  # cmp abs,x
            rel = d[o + 7] - 256 if d[o + 7] >= 128 else d[o + 7]
            cands.append((o, rel))
    cands.sort()
    return cands


def _detect_optimizations(img):
    """Recover (PULSEOPTIMIZATION, REALTIMEOPTIMIZATION) from the player image.

    PULSEOPTIMIZATION: with the pulse skip on, the gateoff-timer check sits at
    ``mt_done`` *before* the pulse-execution block (player.s ~1077-1087), so its
    ``beq mt_getnewnote`` must branch forward over the whole pulse routine (a
    large positive displacement).  With the optimization off the same check sits
    *after* the pulse block (player.s ~1194), and ``mt_getnewnote`` is only a few
    bytes away (small displacement).  The gateoff check is the middle of the
    three reload/gateoff ``BD..;cmp;F0`` sites.

    REALTIMEOPTIMIZATION: adds a ``lda mt_chncounter,x ; beq mt_gatetimer``
    (``BD lo hi ; F0 rel``, no cmp) tick-0 skip at ``mt_wavedone`` (player.s
    ~972-978).  Without the optimization that extra ghostreg test is absent, so
    the count of pure ``lda <ramreg>,x ; beq`` sites drops from three to two."""
    d = img.data
    cands = _gateoff_candidates(img)
    pulseopt = 0
    if len(cands) >= 3:
        mid = cands[len(cands) // 2]
        pulseopt = 1 if mid[1] > 40 else 0
    elif cands:
        pulseopt = 1 if max(c[1] for c in cands) > 40 else 0
    # pure lda <ramreg>,x ; beq  (the reg's hi byte is RAM, never $D4xx SID)
    rt_sites = sum(
        1
        for o in range(len(d) - 4)
        if d[o] == 0xBD and d[o + 3] == 0xF0 and d[o + 2] != 0xD4
    )
    realtimeopt = 1 if rt_sites >= 3 else 0
    return bool(pulseopt), bool(realtimeopt)


def _detect_simplepulse(img, lay):
    """True iff the build set SIMPLEPULSE=1 (greloc's one-byte pulse table).

    greloc's SIMPLEPULSE optimization (greloc.c ~888/1302) folds a pulse-table
    SET step into ONE packed byte ``(pulsehi & 0x0f) | (pulselo & 0xf0)`` and
    pre-swaps the modulation speed (``swapnybbles``).  The packed player then
    keeps a SINGLE ghost pulse byte: mt_setpulse stores it to both lo and hi
    (no separate hi store), and mt_pulsemod (player.s ~1148-1164,
    ``.IF SIMPLEPULSE != 0``) does an 8-bit ``lo = lo + speed + carry``
    accumulate:

        lda <ghostpulselo,x / lda mt_chnpulselo,x   ; the running lowbyte
        clc
        adc mt_pulsespdtbl-1,y   (79 lo hi)         ; + packed speed
        adc #$00                 (69 00)            ; + carry-out (fold back)
        sta ...pulselo / sta ...pulsehi

    The signature is ``adc mt_pulsespdtbl-1,y`` (absolute,y add of the
    pulse-SPEED table, operand == ``pulse_R - 1``) immediately followed by
    ``adc #$00``.  Anchoring on the pulse-speed-table operand is essential: the
    bare ``79 .. 69 00`` byte pattern also occurs in unrelated 16-bit pointer
    adds elsewhere in some players (e.g. Shadow_over_Innsmouth, a full-mod
    build), so a pattern-only match false-positives.  The full-mod player's
    pulse modulation instead uses ``dec/inc chnpulsehi`` 16-bit arithmetic and
    never adds the pulse-speed table with a trailing ``adc #$00``.

    NOPULSEMOD-only builds (no modulation entries) without SIMPLEPULSE have no
    such add and render identically to the editor, so they need no flag.

    The ``adc #$00`` anchor only fires when the build HAS a pulse-MOD step.  A
    SIMPLEPULSE build whose pulse table is all SET steps (no modulation) carries
    no ``mt_pulsemod`` accumulate, so we additionally anchor on the SIMPLEPULSE
    SET-PULSE step itself.  In ``mt_setpulse`` (player.s ~1108-1124) the editor
    (``SIMPLEPULSE == 0``) stores the pulse HIGH byte from the value still in A
    (the ``mt_pulsetimetbl`` byte that selected the set step) BEFORE loading the
    speed/low byte -- ``sta ghostpulsehi ; lda mt_pulsespdtbl-1,y ; sta
    ghostpulselo`` -- so a store sits between the time-table read and the
    speed-table read.  The SIMPLEPULSE build (``SIMPLEPULSE != 0``) drops that
    high store and feeds the SPEED byte to both registers, so the speed-table
    load immediately follows the time-table load with NO store between:

        lda mt_pulsetimetbl-1,y   ; B9 (pulse_L-1)lo (pulse_L-1)hi   (set-step test)
        lda mt_pulsespdtbl-1,y    ; B9 (pulse_R-1)lo (pulse_R-1)hi   (-> A = speed)
        sta ghostpulselo,x        ; then stored to BOTH lo and hi

    The signature is therefore two back-to-back ``lda ...,y`` of the pulse TIME
    table then the pulse SPEED table (``B9 Llo Lhi B9 Rlo Rhi``).  The full-mod
    set step never has two consecutive Y-indexed loads here (its high store
    breaks them up), so this never false-positives on full-mod builds (verified:
    0 hits across the byte-exact corpus, incl. Jetta/Hammurabi/Truck-On).
    """
    d = img.data
    speedtbl = (lay["pulse_R"] - 1) & 0xFFFF  # mt_pulsespdtbl-1 operand
    lo, hi = speedtbl & 0xFF, (speedtbl >> 8) & 0xFF
    for o in range(len(d) - 5):
        if (
            d[o] == 0x79  # adc abs,y
            and d[o + 1] == lo
            and d[o + 2] == hi
            and d[o + 3] == 0x69  # adc #imm
            and d[o + 4] == 0x00
        ):
            return True
    # SET-PULSE-only SIMPLEPULSE: 'lda pulsetimetbl-1,y' directly followed by
    # 'lda pulsespdtbl-1,y' (no intervening high-byte store).
    timetbl = (lay["pulse_L"] - 1) & 0xFFFF  # mt_pulsetimetbl-1 operand
    tlo, thi = timetbl & 0xFF, (timetbl >> 8) & 0xFF
    for o in range(len(d) - 6):
        if (
            d[o] == 0xB9  # lda mt_pulsetimetbl-1,y
            and d[o + 1] == tlo
            and d[o + 2] == thi
            and d[o + 3] == 0xB9  # lda mt_pulsespdtbl-1,y
            and d[o + 4] == lo
            and d[o + 5] == hi
        ):
            return True
    return False


def _vibparam_base(img, lay):
    """Address of ``mt_insvibparam`` (the instrument vibrato-param column,
    instrument 1 at offset 0), or None when the build has no instrument vibrato
    (NOINSTRVIB, so the column is absent)."""
    flags = _instrument_flags(img, lay)
    if flags["noinsvib"]:
        return None
    cols = lay["cols"]
    ci = 3  # ad, sr, wave
    if not flags["nopulse"]:
        ci += 1
    if not flags["nofilter"]:
        ci += 1
    return cols[ci]  # vibparam column base == mt_insvibparam


def _detect_live_vibrato(img, lay):
    """True iff the build reads the instrument vibrato param LIVE every tick.

    greloc's instrument-vibrato-only build (NOEFFECTS != 0: no pattern-FX
    machinery -- no ``mt_chnfx``/``mt_chnparam``/effect jump table) runs the
    continuous instrument vibrato straight off the channel's CURRENT instrument
    each tick (player.s ``mt_wavedone`` .ELSE branch):

        ldy mt_chninstr,y          ; current channel instrument
        lda mt_insvibparam-1,y     ; B9  (vibparam-1) -- its speed-table param
        tay                        ; A8

    so at a gate-off boundary -- where ``mt_getnewnote`` flips ``mt_chninstr``
    to the new note's instrument 1-2 frames before the note inits -- the held
    (old) frequency is vibrated with the NEW instrument's param. The full-FX
    build (NOEFFECTS == 0) instead dispatches through ``mt_chnparam``, a param
    LATCHED only at note-init, so it does NOT vibrate during that window.

    The signature is the live read ``B9 (vibparam-1)lo (vibparam-1)hi A8``: an
    ``lda mt_insvibparam-1,y`` immediately followed by ``tay`` (the
    continuous-effect path consumes the param into Y). The full-FX build's
    only ``lda mt_insvibparam-1,y`` is in note-init and is followed by
    ``sta mt_chnparam,x`` (95 ..), never ``tay``."""
    base = _vibparam_base(img, lay)
    if base is None:
        return False
    operand = (base - 1) & 0xFFFF
    lo, hi = operand & 0xFF, (operand >> 8) & 0xFF
    d = img.data
    for o in range(len(d) - 3):
        if d[o] == 0xB9 and d[o + 1] == lo and d[o + 2] == hi and d[o + 3] == 0xA8:
            return True
    return False


# =========================== packed freq table =============================
def _packed_freq_table(img, lay):
    """The 128-entry note->frequency table the PACKED player actually reads.

    The editor playroutine (gplay.c, mirrored by pygoattracker's
    ``constants.FREQ_TABLE``) reads a fixed table zero-padded to 128 entries.
    The packed player (gt2reloc/player.s) instead indexes
    ``mt_freqtbllo-FIRSTNOTE,y`` / ``mt_freqtblhi-FIRSTNOTE,y`` where the image
    lays the table out UNPADDED as ``freqtbllo[firstnote..lastnote]`` followed
    immediately by ``freqtblhi[...]`` then ``mt_songtbllo``.  A note pushed out
    of range by a wavetable relative step (``note &= $7f`` reaching 96..127, or
    a low ``firstnote`` letting a high note exceed ``lastnote``) therefore
    overruns the unpadded table into the adjacent relocated bytes and produces
    a frequency that is a deterministic function of the packer's relocation
    layout -- the overrun lo bytes are the CONSTANT standard ``freqtblhi``
    table, and the hi bytes are the low bytes of the relocated orderlist /
    pattern absolute addresses -- whereas the editor table returns 0.  (Not
    irreducible: traced byte-for-byte to source.)

    Reconstruct the exact bytes the packed player reads, straight from the
    image: for absolute note ``n`` (0..127),
        lo = img[freqtbllo - firstnote + n]
        hi = img[freqtbllo + L - firstnote + n]
    For in-range notes this reproduces the standard table; out-of-range it
    reads the real overrun bytes.  Addresses outside the image read 0 (the
    note never reaches them in a valid song)."""
    fl = lay["freqtbllo"]
    L = lay["L"]
    fn = lay["firstnote"]
    lo_base = fl - fn  # address of mt_freqtbllo - FIRSTNOTE
    hi_base = fl + L - fn  # address of mt_freqtblhi - FIRSTNOTE
    end = img.end()
    table = []
    for n in range(128):
        lo_a, hi_a = lo_base + n, hi_base + n
        lo = img[lo_a] if img.load <= lo_a < end else 0
        hi = img[hi_a] if img.load <= hi_a < end else 0
        table.append((hi << 8) | lo)
    return tuple(table)


# =========================== public API ====================================
def _build_song(img, lay, flags, name):
    wave, pulse, filt, speed = _build_tables(img, lay)
    return Song(
        name=name,
        subtunes=_build_orderlists(img, lay),
        instruments=_build_instruments(img, lay, flags),
        patterns=_build_patterns(img, lay),
        wavetable=wave,
        pulsetable=pulse,
        filtertable=filt,
        speedtable=speed,
    )


# module-level state shared with render (adparam + last recovered flags)
ADPARAM = 0xFF00
FLAGS = {}
OPTIMIZE_PULSE = False
OPTIMIZE_REALTIME = False
FREQ_TABLE = None  # packed-image note->freq table (None -> editor's table)
SIMPLEPULSE = False  # packed SIMPLEPULSE one-byte pulse build flag
LIVE_VIBRATO = False  # NOEFFECTS!=0 build: live instrument-vibrato param read


def reconstruct_song(sid_path=SID_PATH):
    """Auto-derive the packed layout and return a pygoattracker ``Song``.

    When the instrument-flag disambiguation is uncertain it reconstructs under
    each candidate and keeps the one whose render matches the real player.
    """
    global ADPARAM, FLAGS, OPTIMIZE_PULSE, OPTIMIZE_REALTIME, FREQ_TABLE
    global SIMPLEPULSE, LIVE_VIBRATO
    img = _Image(sid_path)
    lay = _derive_layout(img)
    # Recover the hard-restart ADPARAM/SRPARAM from the player image (the
    # immediates the chip actually executes, player.s 1303/1313) rather than
    # the first AD/SR write the emulator emits -- that heuristic captures the
    # first NOTE's ADSR when no hard-restart frame precedes it, mis-recovering
    # the HR value and holding a stale ADSR through the gate-off frames.
    ADPARAM = _adparam_from_image(img) or _adparam_from_py65(sid_path) or 0x0F00
    # The pulse/realtime skip optimizations are baked into the player image at
    # pack time; derive them per-SID so the render matches this build exactly.
    OPTIMIZE_PULSE, OPTIMIZE_REALTIME = _detect_optimizations(img)
    # SIMPLEPULSE folds the pulse-hi nibble into the one packed pulse byte and
    # the packed player computes pulse width by a different code path; detect it
    # so the render runs that path on the (un-inverted) packed pulse table.
    SIMPLEPULSE = _detect_simplepulse(img, lay)
    # NOEFFECTS!=0 build (instrument-vibrato-only): the continuous vibrato reads
    # its param LIVE from the current channel instrument, so it vibrates the held
    # frequency through the 1-2 gate-off frames before a new note inits. Detect it
    # so the render reproduces that gate-off-window modulation.
    LIVE_VIBRATO = _detect_live_vibrato(img, lay)
    # The packed player reads an UNPADDED freq table; recover the exact 128-entry
    # table (overrun bytes included) so out-of-range notes render byte-exactly.
    FREQ_TABLE = _packed_freq_table(img, lay)

    flags = _instrument_flags(img, lay)
    name = sid_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    candidates = [flags]
    # If the extra-pair role is genuinely ambiguous (no chnwave signature to
    # check against), try both vib-only and gate-only and pick by render match.
    ncols = lay["ncols"]
    base = 3 + (0 if flags["nopulse"] else 1) + (0 if flags["nofilter"] else 1)
    if (ncols - base) == 2 and _chnwave_reg(img, lay["wave_L"]) is None:
        candidates = [
            dict(flags, noinsvib=0, fixedparams=1),
            dict(flags, noinsvib=1, fixedparams=0),
        ]

    if len(candidates) == 1:
        FLAGS = _flag_summary(img, lay, candidates[0])
        return _build_song(img, lay, candidates[0], name)

    ref = render_py65(sid_path, 600)
    best = None
    for cand in candidates:
        song = _build_song(img, lay, cand, name)
        try:
            got = render_state(song, 600)
        except Exception:
            continue
        _, ok = best_lag(got, ref)
        if best is None or ok > best[0]:
            best = (ok, cand, song)
    FLAGS = _flag_summary(img, lay, best[1])
    return best[2]


def render_params():
    """The per-SID render parameters set by the last reconstruct_song call."""
    return {
        "adparam": ADPARAM,
        "optimize_pulse": OPTIMIZE_PULSE,
        "optimize_realtime": OPTIMIZE_REALTIME,
        "freq_table": FREQ_TABLE,
        "simplepulse": SIMPLEPULSE,
        "live_vibrato": LIVE_VIBRATO,
    }


def _flag_summary(img, lay, flags):
    firstnohr, firstlegato = _recover_hr_order(img)
    return dict(
        songs=lay["songs"],
        instruments=lay["I"],
        patterns=lay["P"],
        firstnote=lay["firstnote"],
        lastnote=lay["lastnote"],
        nopulse=flags["nopulse"],
        nofilter=flags["nofilter"],
        noinsvib=flags["noinsvib"],
        fixedparams=flags["fixedparams"],
        FIRSTNOHRINSTR=firstnohr,
        FIRSTLEGATOINSTR=firstlegato,
        wave_len=lay["wl"],
        pulse_len=lay["pl"],
        filter_len=lay["fl"],
        speed_len=lay["sl"],
        speedzero=lay["speedzero"],
        ncols=lay["ncols"],
        optimize_pulse=OPTIMIZE_PULSE,
        optimize_realtime=OPTIMIZE_REALTIME,
        simplepulse=SIMPLEPULSE,
        live_vibrato=LIVE_VIBRATO,
    )


# =========================== rendering =====================================
def _mask(reg):
    r = list(reg)
    for i in _PW_HI:
        r[i] &= 0x0F
    return r


_UNSET = object()


def render_state(
    song,
    nframes,
    adparam=None,
    optimize_pulse=None,
    optimize_realtime=None,
    subtune=0,
    freq_table=_UNSET,
    simplepulse=None,
    live_vibrato=None,
):
    """Per-frame 25-register SID state (forward-held), aligned 1:1 with the
    real player.  pygoattracker folds song-init into the first play_frame call
    (which emits no register writes); we render nframes+1 and drop that leading
    INIT frame so frame 0 is the first audible frame, matching py65/VICE.

    The per-SID render parameters (adparam, optimize_pulse/realtime, freq_table)
    default to the module globals set by the last reconstruct_song; pass them
    explicitly to render a recovered program without that global state.
    ``freq_table`` is the packed-image note->frequency table (so out-of-range
    notes reproduce the packed player's overrun frequencies); ``None`` falls
    back to pygoattracker's editor table.  ``simplepulse`` selects the packed
    SIMPLEPULSE pulse path (one packed byte fed to both pulse-lo and pulse-hi);
    ``None`` uses the last-detected flag, ``False`` the editor pulse path."""
    if adparam is None:
        adparam = ADPARAM
    if optimize_pulse is None:
        optimize_pulse = OPTIMIZE_PULSE
    if optimize_realtime is None:
        optimize_realtime = OPTIMIZE_REALTIME
    if freq_table is _UNSET:
        freq_table = FREQ_TABLE
    if simplepulse is None:
        simplepulse = SIMPLEPULSE
    if live_vibrato is None:
        live_vibrato = LIVE_VIBRATO
    p = Player(
        song,
        subtune=subtune,
        adparam=adparam,
        optimize_pulse=optimize_pulse,
        optimize_realtime=optimize_realtime,
        freq_table=freq_table,
        simplepulse=bool(simplepulse),
        live_vibrato=bool(live_vibrato),
    )
    regs = [0] * NREG
    out = np.zeros((nframes, NREG), dtype=np.int64)
    # frame 0 is the INIT frame (no writes); skip it.
    for r, v in p.play_frame():
        if r < NREG:
            regs[r] = v
    for f in range(nframes):
        for r, v in p.play_frame():
            if r < NREG:
                regs[r] = v
        out[f] = _mask(regs)
    return out


# Backwards-compatible alias (a prior revision named it render_song).
render_song = render_state


def render_py65(sid_path, nframes):
    emu = SIDEmu(load_psid(sid_path))
    emu.init(0)
    out = np.zeros((nframes, NREG), dtype=np.int64)
    for f in range(nframes):
        emu.play_frame()
        out[f] = _mask(emu.state())
    return out


def best_lag(a, b, span=12):
    best = (-1, 0)
    n = min(len(a), len(b))
    for lag in range(-span, span + 1):
        ok = cnt = 0
        for i in range(n):
            j = i + lag
            if 0 <= j < n:
                cnt += 1
                ok += int(np.array_equal(a[i], b[j]))
        if cnt > 20 and ok > best[0]:
            best = (ok, lag)
    return best[1], best[0]
