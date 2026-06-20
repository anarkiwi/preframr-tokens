"""PSID loader + py65 6502 with SID-write / RAM instrumentation.

Single-speed PSID/RSID. ``init(subtune)`` once, then ``play_frame`` per PAL
frame; ``state`` snapshots the 25-register SID state (last written value per
register, forward-held) the way ``lane_grammar.per_frame_state`` reconstructs it
from a dump.
"""

import struct
from dataclasses import dataclass

from py65.devices.mpu6502 import MPU

SID_BASE = 0xD400
NREG = 25
_RET = 0xFFFE


@dataclass
class PSID:
    path: str
    version: int
    init_addr: int
    play_addr: int
    songs: int
    start_song: int
    speed: int
    flags: int
    load_addr: int
    data: bytes


def load_psid(path):
    """Parse a PSID/RSID header + program image."""
    b = open(path, "rb").read()
    assert b[0:4] in (b"PSID", b"RSID"), b[0:4]
    ver = struct.unpack(">H", b[4:6])[0]
    data_off = struct.unpack(">H", b[6:8])[0]
    load_addr = struct.unpack(">H", b[8:10])[0]
    init_addr = struct.unpack(">H", b[10:12])[0]
    play_addr = struct.unpack(">H", b[12:14])[0]
    songs = struct.unpack(">H", b[14:16])[0]
    start = struct.unpack(">H", b[16:18])[0]
    speed = struct.unpack(">I", b[18:22])[0]
    flags = struct.unpack(">H", b[0x76:0x78])[0] if ver >= 2 else 0
    data = b[data_off:]
    if load_addr == 0:
        load_addr = struct.unpack("<H", data[0:2])[0]
        data = data[2:]
    return PSID(
        path, ver, init_addr, play_addr, songs, start, speed, flags, load_addr, data
    )


class _Mem(list):
    """64K RAM as ints with a per-frame SID-write log."""

    def __init__(self):
        super().__init__([0] * 0x10000)
        self.log = False
        self.sid_writes = []
        self.watch = set()
        self.watch_hits = []

    def __setitem__(self, addr, val):
        if isinstance(addr, slice):
            return super().__setitem__(addr, val)
        if self.log:
            if SID_BASE <= addr < SID_BASE + 0x20:
                self.sid_writes.append((addr - SID_BASE, val & 0xFF))
            elif addr in self.watch:
                self.watch_hits.append((addr, val & 0xFF))
        return super().__setitem__(addr, val & 0xFF)


class SIDEmu:
    """Run a PSID's init/play under py65, snapshotting SID state per frame."""

    def __init__(self, psid):
        self.psid = psid
        self.mem = _Mem()
        for i, byte in enumerate(psid.data):
            self.mem[psid.load_addr + i] = byte
        self.mpu = MPU(memory=self.mem)

    def _call(self, addr, a=0, max_steps=4_000_000):
        mpu = self.mpu
        mpu.a, mpu.x, mpu.y = a & 0xFF, 0, 0
        mpu.stPushWord(_RET - 1)
        mpu.pc = addr
        steps = 0
        while steps < max_steps and mpu.pc != _RET:
            mpu.step()
            steps += 1
        return steps

    def init(self, subtune):
        self.mem.log = True
        self._call(self.psid.init_addr, a=subtune)
        self.mem.log = False

    def play_frame(self):
        """Run one play call; return its (reg, val) SID writes in order."""
        self.mem.log = True
        self.mem.sid_writes = []
        self.mem.watch_hits = []
        self._call(self.psid.play_addr)
        self.mem.log = False
        return self.mem.sid_writes

    def state(self):
        """The 25-register SID state (last written value per register)."""
        return [self.mem[SID_BASE + r] for r in range(NREG)]
