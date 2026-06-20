"""Unit tests for the undocumented-opcode 6502 (``MPU65ILL``).

lft's relocated player uses NMOS-undocumented opcodes (LAX/SAX/ALR/ANC/ARR/AXS/
SLO/SRE/RLA/RRA/DCP/ISC + the multi-byte illegal NOPs); py65's base MPU returns
"not implemented" for them, so the white-box recovery cannot run without these.
Each opcode is driven once here (covering both addressing-mode helpers and the
NMOS flag semantics) so the implementation is exercised independent of any one
tune. Expected values are the canonical NMOS behaviours documented in the source.
"""

import pytest

from preframr_tokens.bacc.illegal_mpu import MPU65ILL


def _cpu():
    mpu = MPU65ILL(memory=[0] * 0x10000)
    mpu.pc = 0x1000
    return mpu


def _run(mpu, opcode, operands=(), setup=None):
    """Place opcode+operands at PC, optionally run ``setup(mpu)``, step once."""
    if setup:
        setup(mpu)
    addr = mpu.pc
    mpu.memory[addr] = opcode
    for i, b in enumerate(operands):
        mpu.memory[addr + 1 + i] = b
    mpu.step()


# ---- LAX (A = X = M) across every addressing mode ----------------------------
def test_lax_zeropage():
    mpu = _cpu()
    mpu.memory[0x0040] = 0x37
    _run(mpu, 0xA7, (0x40,))
    assert mpu.a == 0x37 and mpu.x == 0x37


def test_lax_zeropage_y():
    mpu = _cpu()
    mpu.y = 2
    mpu.memory[0x0042] = 0x11
    _run(mpu, 0xB7, (0x40,))
    assert mpu.a == 0x11 and mpu.x == 0x11


def test_lax_absolute_sets_zero_flag():
    mpu = _cpu()
    mpu.memory[0x2000] = 0x00
    _run(mpu, 0xAF, (0x00, 0x20))
    assert mpu.a == 0 and (mpu.p & mpu.ZERO)


def test_lax_absolute_y_sets_negative_flag():
    mpu = _cpu()
    mpu.y = 1
    mpu.memory[0x2001] = 0x80
    _run(mpu, 0xBF, (0x00, 0x20))
    assert mpu.a == 0x80 and (mpu.p & mpu.NEGATIVE)


def test_lax_indirect_y():
    mpu = _cpu()
    mpu.y = 3
    mpu.memory[0x0010] = 0x00
    mpu.memory[0x0011] = 0x30
    mpu.memory[0x3003] = 0x5A
    _run(mpu, 0xB3, (0x10,))
    assert mpu.a == 0x5A and mpu.x == 0x5A


def test_lax_indirect_x():
    mpu = _cpu()
    mpu.x = 4
    mpu.memory[0x0014] = 0x00
    mpu.memory[0x0015] = 0x40
    mpu.memory[0x4000] = 0x99
    _run(mpu, 0xA3, (0x10,))
    assert mpu.a == 0x99 and mpu.x == 0x99


def test_lax_immediate():
    mpu = _cpu()
    _run(mpu, 0xAB, (0x7E,))
    assert mpu.a == 0x7E and mpu.x == 0x7E


# ---- SAX (M = A & X) ---------------------------------------------------------
def test_sax_zeropage():
    mpu = _cpu()
    mpu.a, mpu.x = 0xF0, 0x3C
    _run(mpu, 0x87, (0x20,))
    assert mpu.memory[0x0020] == (0xF0 & 0x3C)


def test_sax_zeropage_y():
    mpu = _cpu()
    mpu.a, mpu.x, mpu.y = 0xFF, 0x0F, 1
    _run(mpu, 0x97, (0x20,))
    assert mpu.memory[0x0021] == 0x0F


def test_sax_absolute():
    mpu = _cpu()
    mpu.a, mpu.x = 0xCC, 0xAA
    _run(mpu, 0x8F, (0x00, 0x21))
    assert mpu.memory[0x2100] == (0xCC & 0xAA)


def test_sax_indirect_x():
    mpu = _cpu()
    mpu.a, mpu.x = 0xF3, 0x06
    mpu.memory[0x0006] = 0x00
    mpu.memory[0x0007] = 0x22
    _run(mpu, 0x83, (0x00,))
    assert mpu.memory[0x2200] == (0xF3 & 0x06)


# ---- ALR / ANC / ARR / AXS (immediate ALU illegals) --------------------------
def test_alr_shifts_and_sets_carry():
    mpu = _cpu()
    mpu.a = 0xFF
    _run(mpu, 0x4B, (0x03,))  # (0xFF & 3)=3 -> >>1 = 1, carry=bit0=1
    assert mpu.a == 0x01 and (mpu.p & mpu.CARRY)


def test_alr_result_zero():
    mpu = _cpu()
    mpu.a = 0xF0
    _run(mpu, 0x4B, (0x01,))  # (0xF0 & 1)=0 -> 0, carry=0, zero set
    assert mpu.a == 0 and (mpu.p & mpu.ZERO) and not (mpu.p & mpu.CARRY)


@pytest.mark.parametrize("opcode", [0x0B, 0x2B])
def test_anc_carry_from_bit7(opcode):
    mpu = _cpu()
    mpu.a = 0xFF
    _run(mpu, opcode, (0x80,))
    assert mpu.a == 0x80 and (mpu.p & mpu.CARRY) and (mpu.p & mpu.NEGATIVE)


def test_anc_clears_carry_when_bit7_clear():
    mpu = _cpu()
    mpu.a = 0x7F
    _run(mpu, 0x0B, (0x0F,))
    assert mpu.a == 0x0F and not (mpu.p & mpu.CARRY)


def test_arr_negative_when_carry_in():
    mpu = _cpu()
    mpu.a = 0xFF
    mpu.p |= mpu.CARRY  # c_in -> bit7 of result set => negative
    _run(mpu, 0x6B, (0xC0,))  # v=0xC0; res=(0xC0>>1)|0x80 = 0xE0
    assert mpu.a == 0xE0 and (mpu.p & mpu.NEGATIVE) and (mpu.p & mpu.CARRY)


def test_arr_carry_and_overflow():
    mpu = _cpu()
    mpu.a = 0xFF
    mpu.p &= ~mpu.CARRY  # no c_in -> bit7 of result clear
    _run(mpu, 0x6B, (0x80,))  # v=0x80; res=0x40: bit6=1 (carry), bit6^bit5=1 (V)
    assert mpu.a == 0x40 and (mpu.p & mpu.CARRY) and (mpu.p & mpu.OVERFLOW)


def test_arr_result_zero():
    mpu = _cpu()
    mpu.a = 0x00
    mpu.p &= ~mpu.CARRY
    _run(mpu, 0x6B, (0xFF,))
    assert mpu.a == 0 and (mpu.p & mpu.ZERO)


def test_axs_subtracts_into_x_with_borrow():
    mpu = _cpu()
    mpu.a, mpu.x = 0xF0, 0x0F  # A&X = 0
    _run(mpu, 0xCB, (0x01,))  # 0 - 1 = -1 -> 0xFF, no carry (borrow), negative
    assert mpu.x == 0xFF and not (mpu.p & mpu.CARRY) and (mpu.p & mpu.NEGATIVE)


def test_axs_no_borrow_sets_carry_and_zero():
    mpu = _cpu()
    mpu.a, mpu.x = 0xFF, 0x05  # A&X = 5
    _run(mpu, 0xCB, (0x05,))  # 5 - 5 = 0 -> carry set, zero set
    assert mpu.x == 0 and (mpu.p & mpu.CARRY) and (mpu.p & mpu.ZERO)


# ---- SLO / SRE / RLA / RRA / DCP / ISC (read-modify-write illegals) ----------
def test_slo_zeropage_shifts_mem_and_ors_a():
    mpu = _cpu()
    mpu.a = 0x01
    mpu.memory[0x0030] = 0x81  # <<1 = 0x02, carry from bit7
    _run(mpu, 0x07, (0x30,))
    assert mpu.memory[0x0030] == 0x02 and mpu.a == 0x03 and (mpu.p & mpu.CARRY)


def test_slo_addressing_modes_cover():
    # zpx, abs, abx, aby, inx, iny -- each just needs to execute its addr helper.
    for opcode, ops, setup in (
        (0x17, (0x30,), lambda m: setattr(m, "x", 1)),
        (0x0F, (0x00, 0x40), None),
        (0x1F, (0x00, 0x40), lambda m: setattr(m, "x", 1)),
        (0x1B, (0x00, 0x40), lambda m: setattr(m, "y", 1)),
        (0x03, (0x10,), _ptr_x),
        (0x13, (0x10,), _ptr_y),
    ):
        mpu = _cpu()
        _run(mpu, opcode, ops, setup)


def _ptr_x(m):
    m.x = 2
    m.memory[0x0012] = 0x00
    m.memory[0x0013] = 0x45


def _ptr_y(m):
    m.y = 2
    m.memory[0x0010] = 0x00
    m.memory[0x0011] = 0x45


def test_sre_zeropage_shifts_mem_and_xors_a():
    mpu = _cpu()
    mpu.a = 0xFF
    mpu.memory[0x0030] = 0x03  # >>1 = 0x01, carry from bit0
    _run(mpu, 0x47, (0x30,))
    assert mpu.memory[0x0030] == 0x01 and mpu.a == (0xFF ^ 0x01)
    assert mpu.p & mpu.CARRY


def test_sre_addressing_modes_cover():
    for opcode, ops, setup in (
        (0x57, (0x30,), lambda m: setattr(m, "x", 1)),
        (0x4F, (0x00, 0x40), None),
    ):
        mpu = _cpu()
        _run(mpu, opcode, ops, setup)


def test_rla_rotates_mem_and_ands_a():
    mpu = _cpu()
    mpu.a = 0xFF
    mpu.p |= mpu.CARRY  # c_in -> rotated into bit0
    mpu.memory[0x0030] = 0x80  # ROL: (0x80<<1)|1 = 0x01, carry from old bit7
    _run(mpu, 0x27, (0x30,))
    assert mpu.memory[0x0030] == 0x01 and mpu.a == 0x01 and (mpu.p & mpu.CARRY)


def test_rla_no_carry_out_when_bit7_clear():
    mpu = _cpu()
    mpu.a = 0xFF
    mpu.p &= ~mpu.CARRY  # no c_in
    mpu.memory[0x0030] = 0x01  # ROL: (1<<1)|0 = 0x02, old bit7 clear -> carry clear
    _run(mpu, 0x27, (0x30,))
    assert mpu.memory[0x0030] == 0x02 and mpu.a == 0x02 and not (mpu.p & mpu.CARRY)


def test_rra_rotates_mem_then_adcs():
    mpu = _cpu()
    mpu.a = 0x10
    mpu.p &= ~mpu.CARRY
    mpu.memory[0x0030] = 0x02  # ROR -> 0x01 (newc=0); A = 0x10 + 0x01 = 0x11
    _run(mpu, 0x67, (0x30,))
    assert mpu.memory[0x0030] == 0x01 and mpu.a == 0x11


def test_rra_overflow_and_negative():
    mpu = _cpu()
    mpu.a = 0x40
    mpu.p |= mpu.CARRY  # ROR pulls carry into bit7 -> m = 0x80
    mpu.memory[0x0030] = 0x00  # ROR with c_in -> 0x80; A = 0x40 + 0x80 = 0xC0
    _run(mpu, 0x67, (0x30,))
    # 0x40 + 0x80: same-sign operands (both -> bit7 differs)... 0x40 ^ 0x80 = 0xC0
    # & 0x80 set, so NO overflow per the impl; result 0xC0 is negative.
    assert mpu.a == 0xC0 and (mpu.p & mpu.NEGATIVE) and not (mpu.p & mpu.OVERFLOW)


def test_rra_overflow_set():
    mpu = _cpu()
    mpu.a = 0x40
    mpu.p &= ~mpu.CARRY
    mpu.memory[0x0030] = 0x80  # ROR (no c_in) -> 0x40, newc=0; 0x40+0x40 = 0x80
    _run(mpu, 0x67, (0x30,))
    # 0x40 ^ 0x40 = 0 (same sign); 0x40 ^ 0x80 = 0xC0 & 0x80 set -> overflow
    assert mpu.a == 0x80 and (mpu.p & mpu.OVERFLOW)


def test_rra_carry_out():
    mpu = _cpu()
    mpu.a = 0xFF
    mpu.p &= ~mpu.CARRY
    mpu.memory[0x0030] = 0x02  # ROR -> 0x01; A = 0xFF + 1 = 0x100 -> carry, zero
    _run(mpu, 0x67, (0x30,))
    assert mpu.a == 0x00 and (mpu.p & mpu.CARRY) and (mpu.p & mpu.ZERO)


def test_dcp_decrements_mem_and_compares():
    mpu = _cpu()
    mpu.a = 0x05
    mpu.memory[0x0030] = 0x06  # -- -> 0x05; A - m = 0 -> carry+zero
    _run(mpu, 0xC7, (0x30,))
    assert mpu.memory[0x0030] == 0x05 and (mpu.p & mpu.CARRY) and (mpu.p & mpu.ZERO)


def test_dcp_borrow_negative():
    mpu = _cpu()
    mpu.a = 0x00
    mpu.memory[0x0030] = 0x02  # -- -> 0x01; A - m = -1 -> no carry, negative
    _run(mpu, 0xC7, (0x30,))
    assert not (mpu.p & mpu.CARRY) and (mpu.p & mpu.NEGATIVE)


def test_isc_increments_mem_then_sbc():
    mpu = _cpu()
    mpu.a = 0x05
    mpu.p |= mpu.CARRY  # no borrow
    mpu.memory[0x0030] = 0x01  # ++ -> 0x02; A - 2 = 3
    _run(mpu, 0xE7, (0x30,))
    assert mpu.memory[0x0030] == 0x02 and mpu.a == 0x03 and (mpu.p & mpu.CARRY)


def test_isc_overflow_path():
    mpu = _cpu()
    mpu.a = 0x80
    mpu.p |= mpu.CARRY
    mpu.memory[0x0030] = 0x00  # ++ -> 0x01; 0x80 - 1 = 0x7F, signed overflow
    _run(mpu, 0xE7, (0x30,))
    assert mpu.a == 0x7F and (mpu.p & mpu.OVERFLOW)


def test_isc_borrow_clears_carry():
    mpu = _cpu()
    mpu.a = 0x01
    mpu.p |= mpu.CARRY  # start with no borrow
    mpu.memory[0x0030] = 0x04  # ++ -> 0x05; 0x01 - 0x05 = -4 -> borrow (carry clear)
    _run(mpu, 0xE7, (0x30,))
    assert mpu.a == 0xFC and not (mpu.p & mpu.CARRY) and (mpu.p & mpu.NEGATIVE)


def test_isc_zero_result():
    mpu = _cpu()
    mpu.a = 0x02
    mpu.p |= mpu.CARRY
    mpu.memory[0x0030] = 0x01  # ++ -> 0x02; 2 - 2 = 0
    _run(mpu, 0xE7, (0x30,))
    assert mpu.a == 0 and (mpu.p & mpu.ZERO)


# ---- illegal multi-byte NOPs -------------------------------------------------
def test_illegal_nops_advance_pc():
    for opcode, length in ((0x80, 2), (0x04, 2), (0x14, 2), (0x1C, 3)):
        mpu = _cpu()
        start = mpu.pc
        if opcode in (0x14, 0x1C):
            mpu.x = 1
        _run(mpu, opcode, (0x00, 0x20))
        assert mpu.pc == start + length
