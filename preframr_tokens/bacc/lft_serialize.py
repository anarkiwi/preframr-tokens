"""Serialize an lft (algorithmic-RSID) BaccProgram to/from token ids.

The lft recovery is a PURE GENERATOR: the program is the tiny static program
image (the ~254-byte player + embedded seed) plus the 25-register boot frame and
a frame count. There is no per-frame score, no note table, and no instrument
table -- the player itself regenerates every frame from one CIA-driven counter
(see ``backends/lft.py``). The token form therefore carries only:

  header : nframes, load_addr, init_addr, boot[25]
  image  : length, image[]  -- the program image (the seed = the whole tune)

All values ride the same shared base-16 LEB digit alphabet as the other codecs
(no new token ids). The round-trip is exact and ``render`` regenerates the dump
byte-exact (modulo the documented don't-care bits), so this is the extreme
encoding-sparsity case: ~0.03 image-bytes/frame.
"""

from preframr_tokens.bacc.backends.lft import CIA_CPF
from preframr_tokens.bacc.primitive import BaccProgram
from preframr_tokens.bacc.serialize import _ru, _wu

NREG = 25


def lft_program_to_ids(program):
    """Serialize an lft BaccProgram to a flat list of token ids."""
    out = []
    _wu(out, program.nframes)
    _wu(out, program.seed["load_addr"])
    _wu(out, program.seed["init_addr"])
    for b in program.boot:
        _wu(out, b)
    image = program.seed["image"]
    _wu(out, len(image))
    for b in image:
        _wu(out, b)
    return out


def lft_ids_to_program(ids):
    """Inverse of lft_program_to_ids -> BaccProgram."""
    i = 0
    nframes, i = _ru(ids, i)
    load_addr, i = _ru(ids, i)
    init_addr, i = _ru(ids, i)
    boot = []
    for _ in range(NREG):
        b, i = _ru(ids, i)
        boot.append(b)
    n, i = _ru(ids, i)
    image = []
    for _ in range(n):
        b, i = _ru(ids, i)
        image.append(b)
    return BaccProgram(
        driver="lft",
        nframes=nframes,
        boot=boot,
        instruments=[],
        score=[],
        seed={
            "load_addr": load_addr,
            "init_addr": init_addr,
            "image": image,
            "cia_cpf": CIA_CPF,
        },
    )


def lft_measure(program):
    """Return ({block: tokens}, nframes) for the serialized lft program."""
    ids = lft_program_to_ids(program)
    header = len(_wrap(program.nframes)) + len(_wrap(program.seed["load_addr"]))
    header += len(_wrap(program.seed["init_addr"]))
    boot = sum(len(_wrap(b)) for b in program.boot)
    image = len(_wrap(len(program.seed["image"]))) + sum(
        len(_wrap(b)) for b in program.seed["image"]
    )
    brk = {
        "header": header,
        "boot": boot,
        "image": image,
        "total": len(ids),
    }
    return brk, program.nframes


def _wrap(n):
    out = []
    _wu(out, n)
    return out
