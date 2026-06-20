"""MPU6502 subclass adding the undocumented opcodes used by lft's players.
Implements the combined-ALU illegals as their canonical NMOS behavior.
"""

from py65.devices.mpu6502 import MPU
from py65.utils.devices import make_instruction_decorator


class MPU65ILL(MPU):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)


# Build a fresh, copied instruction table so we don't clobber the base class.
MPU65ILL.instruct = MPU.instruct[:]
MPU65ILL.cycletime = list(MPU.cycletime)
MPU65ILL.extracycles = list(MPU.extracycles)
MPU65ILL.disassemble = list(MPU.disassemble)

instruction = make_instruction_decorator(
    MPU65ILL.instruct, MPU65ILL.disassemble, MPU65ILL.cycletime, MPU65ILL.extracycles
)


# ---- helpers ----
def _setNZ(self, val):
    self.p &= ~(self.NEGATIVE | self.ZERO)
    if val == 0:
        self.p |= self.ZERO
    else:
        self.p |= val & self.NEGATIVE


# ---------- LAX: A = X = M ----------
def _LAX(self, addr_fn, plen):
    addr = addr_fn()
    m = self.ByteAt(addr)
    self.a = m
    self.x = m
    _setNZ(self, m)
    self.pc += plen


@instruction(name="LAX", mode="zpg", cycles=3)
def inst_0xa7(self):
    _LAX(self, self.ZeroPageAddr, 1)


@instruction(name="LAX", mode="zpy", cycles=4)
def inst_0xb7(self):
    _LAX(self, self.ZeroPageYAddr, 1)


@instruction(name="LAX", mode="abs", cycles=4)
def inst_0xaf(self):
    _LAX(self, self.AbsoluteAddr, 2)


@instruction(name="LAX", mode="aby", cycles=4)
def inst_0xbf(self):
    _LAX(self, self.AbsoluteYAddr, 2)


@instruction(name="LAX", mode="iny", cycles=5)
def inst_0xb3(self):
    _LAX(self, self.IndirectYAddr, 1)


@instruction(name="LAX", mode="inx", cycles=6)
def inst_0xa3(self):
    _LAX(self, self.IndirectXAddr, 1)


@instruction(name="LAX", mode="imm", cycles=2)
def inst_0xab(self):
    # LXA / ATX: A = X = (A | magic) & imm ; common impl: A=X=imm
    m = self.ByteAt(self.pc)
    self.a = m
    self.x = m
    _setNZ(self, m)
    self.pc += 1


# ---------- SAX: M = A & X ----------
def _SAX(self, addr_fn, plen):
    addr = addr_fn()
    self.memory[addr] = self.a & self.x
    self.pc += plen


@instruction(name="SAX", mode="zpg", cycles=3)
def inst_0x87(self):
    _SAX(self, self.ZeroPageAddr, 1)


@instruction(name="SAX", mode="zpy", cycles=4)
def inst_0x97(self):
    _SAX(self, self.ZeroPageYAddr, 1)


@instruction(name="SAX", mode="abs", cycles=4)
def inst_0x8f(self):
    _SAX(self, self.AbsoluteAddr, 2)


@instruction(name="SAX", mode="inx", cycles=6)
def inst_0x83(self):
    _SAX(self, self.IndirectXAddr, 1)


# ---------- ALR (ASR): A = (A & imm) >> 1, carry = bit0 ----------
@instruction(name="ALR", mode="imm", cycles=2)
def inst_0x4b(self):
    m = self.ByteAt(self.pc)
    self.pc += 1
    v = self.a & m
    self.p &= ~(self.CARRY | self.NEGATIVE | self.ZERO)
    self.p |= v & 1
    v >>= 1
    self.a = v
    if v == 0:
        self.p |= self.ZERO
    # bit7 always 0 after >>1


# ---------- ANC: A = A & imm, carry = bit7 ----------
def _ANC(self):
    m = self.ByteAt(self.pc)
    self.pc += 1
    self.a &= m
    _setNZ(self, self.a)
    self.p &= ~self.CARRY
    if self.a & 0x80:
        self.p |= self.CARRY


@instruction(name="ANC", mode="imm", cycles=2)
def inst_0x0b(self):
    _ANC(self)


@instruction(name="ANC", mode="imm", cycles=2)
def inst_0x2b(self):
    _ANC(self)


# ---------- ARR: A = (A & imm) ROR ; weird flags ----------
@instruction(name="ARR", mode="imm", cycles=2)
def inst_0x6b(self):
    m = self.ByteAt(self.pc)
    self.pc += 1
    v = self.a & m
    c_in = self.p & self.CARRY
    res = (v >> 1) | (0x80 if c_in else 0)
    self.a = res
    self.p &= ~(self.CARRY | self.OVERFLOW | self.NEGATIVE | self.ZERO)
    if res == 0:
        self.p |= self.ZERO
    self.p |= res & self.NEGATIVE
    if res & 0x40:
        self.p |= self.CARRY
    if ((res >> 6) ^ (res >> 5)) & 1:
        self.p |= self.OVERFLOW


# ---------- AXS (SBX): X = (A & X) - imm ; carry like CMP ----------
@instruction(name="AXS", mode="imm", cycles=2)
def inst_0xcb(self):
    m = self.ByteAt(self.pc)
    self.pc += 1
    t = (self.a & self.x) - m
    self.p &= ~(self.CARRY | self.NEGATIVE | self.ZERO)
    if t >= 0:
        self.p |= self.CARRY
    t &= 0xFF
    self.x = t
    if t == 0:
        self.p |= self.ZERO
    self.p |= t & self.NEGATIVE


# ---------- SLO: M <<= 1 ; A |= M ----------
def _SLO(self, addr_fn, plen):
    addr = addr_fn()
    m = self.ByteAt(addr)
    self.p &= ~self.CARRY
    if m & 0x80:
        self.p |= self.CARRY
    m = (m << 1) & 0xFF
    self.memory[addr] = m
    self.a |= m
    _setNZ(self, self.a)
    self.pc += plen


@instruction(name="SLO", mode="zpg", cycles=5)
def inst_0x07(self):
    _SLO(self, self.ZeroPageAddr, 1)


@instruction(name="SLO", mode="zpx", cycles=6)
def inst_0x17(self):
    _SLO(self, self.ZeroPageXAddr, 1)


@instruction(name="SLO", mode="abs", cycles=6)
def inst_0x0f(self):
    _SLO(self, self.AbsoluteAddr, 2)


@instruction(name="SLO", mode="abx", cycles=7)
def inst_0x1f(self):
    _SLO(self, self.AbsoluteXAddr, 2)


@instruction(name="SLO", mode="aby", cycles=7)
def inst_0x1b(self):
    _SLO(self, self.AbsoluteYAddr, 2)


@instruction(name="SLO", mode="inx", cycles=8)
def inst_0x03(self):
    _SLO(self, self.IndirectXAddr, 1)


@instruction(name="SLO", mode="iny", cycles=8)
def inst_0x13(self):
    _SLO(self, self.IndirectYAddr, 1)


# ---------- SRE: M >>= 1 ; A ^= M ----------
def _SRE(self, addr_fn, plen):
    addr = addr_fn()
    m = self.ByteAt(addr)
    self.p &= ~self.CARRY
    if m & 1:
        self.p |= self.CARRY
    m >>= 1
    self.memory[addr] = m
    self.a ^= m
    _setNZ(self, self.a)
    self.pc += plen


@instruction(name="SRE", mode="zpg", cycles=5)
def inst_0x47(self):
    _SRE(self, self.ZeroPageAddr, 1)


@instruction(name="SRE", mode="zpx", cycles=6)
def inst_0x57(self):
    _SRE(self, self.ZeroPageXAddr, 1)


@instruction(name="SRE", mode="abs", cycles=6)
def inst_0x4f(self):
    _SRE(self, self.AbsoluteAddr, 2)


# ---------- RLA: M = ROL M ; A &= M ----------
def _RLA(self, addr_fn, plen):
    addr = addr_fn()
    m = self.ByteAt(addr)
    c_in = self.p & self.CARRY
    self.p &= ~self.CARRY
    if m & 0x80:
        self.p |= self.CARRY
    m = ((m << 1) | (1 if c_in else 0)) & 0xFF
    self.memory[addr] = m
    self.a &= m
    _setNZ(self, self.a)
    self.pc += plen


@instruction(name="RLA", mode="zpg", cycles=5)
def inst_0x27(self):
    _RLA(self, self.ZeroPageAddr, 1)


# ---------- RRA: M = ROR M ; A = A + M + C ----------
def _RRA(self, addr_fn, plen):
    addr = addr_fn()
    m = self.ByteAt(addr)
    c_in = 1 if (self.p & self.CARRY) else 0
    newc = m & 1
    m = (m >> 1) | (0x80 if c_in else 0)
    self.memory[addr] = m
    # ADC m (binary; ignore decimal here, players use CLD)
    result = self.a + m + newc
    self.p &= ~(self.CARRY | self.OVERFLOW | self.NEGATIVE | self.ZERO)
    if ((self.a ^ m) & 0x80) == 0 and ((self.a ^ result) & 0x80):
        self.p |= self.OVERFLOW
    if result > 0xFF:
        self.p |= self.CARRY
    self.a = result & 0xFF
    if self.a == 0:
        self.p |= self.ZERO
    self.p |= self.a & self.NEGATIVE
    self.pc += plen


@instruction(name="RRA", mode="zpg", cycles=5)
def inst_0x67(self):
    _RRA(self, self.ZeroPageAddr, 1)


# ---------- DCP: M-- ; CMP A,M ----------
def _DCP(self, addr_fn, plen):
    addr = addr_fn()
    m = (self.ByteAt(addr) - 1) & 0xFF
    self.memory[addr] = m
    t = self.a - m
    self.p &= ~(self.CARRY | self.NEGATIVE | self.ZERO)
    if t >= 0:
        self.p |= self.CARRY
    t &= 0xFF
    if t == 0:
        self.p |= self.ZERO
    self.p |= t & self.NEGATIVE
    self.pc += plen


@instruction(name="DCP", mode="zpg", cycles=5)
def inst_0xc7(self):
    _DCP(self, self.ZeroPageAddr, 1)


# ---------- ISC: M++ ; SBC ----------
def _ISC(self, addr_fn, plen):
    addr = addr_fn()
    m = (self.ByteAt(addr) + 1) & 0xFF
    self.memory[addr] = m
    data = m
    result = self.a + (~data & 0xFF) + (1 if (self.p & self.CARRY) else 0)
    self.p &= ~(self.CARRY | self.ZERO | self.OVERFLOW | self.NEGATIVE)
    if ((self.a ^ data) & (self.a ^ result)) & 0x80:
        self.p |= self.OVERFLOW
    d = result & 0xFF
    if d == 0:
        self.p |= self.ZERO
    if result > 0xFF:
        self.p |= self.CARRY
    self.p |= d & 0x80
    self.a = d
    self.pc += plen


@instruction(name="ISC", mode="zpg", cycles=5)
def inst_0xe7(self):
    _ISC(self, self.ZeroPageAddr, 1)


# ---------- illegal NOPs (multi-byte) ----------
@instruction(name="NOP", mode="imm", cycles=2)
def inst_0x80(self):
    self.pc += 1


@instruction(name="NOP", mode="zp", cycles=3)
def inst_0x04(self):
    self.pc += 1


@instruction(name="NOP", mode="zpx", cycles=4)
def inst_0x14(self):
    self.pc += 1


@instruction(name="NOP", mode="abx", cycles=4)
def inst_0x1c(self):
    self.AbsoluteXAddr()
    self.pc += 2
