"""Acquire (.sid, .dump.parquet) test fixtures without committing binaries.

The .sid is taken from a local HVSC mirror or downloaded on demand; the register
dump is produced from it by the ``anarkiwi/headlessvice`` VICE container -- the
same ``vsiddump.py`` that built the training corpus -- and cached under
``tests/test_fixtures``. No dumps are committed: the fixture is reproducible from
the .sid alone, byte-for-byte, by anyone with Docker.

Resolution order (sid, then dump): local HVSC path -> cached fixture -> build it
(download the .sid; render the dump in the container). The dump is bounded to the
one subtune under test, using the HVSC Songlengths for the cycle limit.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request

HVSC = os.environ.get("HVSC_ROOT", "/scratch/preframr/hvsc/C64Music")
SONGLENGTHS = os.environ.get(
    "HVSC_SONGLENGTHS", os.path.join(HVSC, "DOCUMENTS", "Songlengths.md5")
)
SONGLENGTHS_URL = os.environ.get(
    "HVSC_SONGLENGTHS_URL",
    "https://hvsc.brona.dk/HVSC/C64Music/DOCUMENTS/Songlengths.md5",
)
IMAGE = os.environ.get("HEADLESSVICE_IMAGE", "anarkiwi/headlessvice")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "test_fixtures")
PAL_PHI = 985248


def _base(sid_path):
    return os.path.basename(sid_path).split(".")[0]


def _songlengths_path():
    """Local HVSC Songlengths.md5, else a cached download (for CI/no-HVSC hosts)."""
    if os.path.exists(SONGLENGTHS):
        return SONGLENGTHS
    cache = os.path.join(FIXTURE_DIR, "Songlengths.md5")
    if not os.path.exists(cache):
        os.makedirs(FIXTURE_DIR, exist_ok=True)
        urllib.request.urlretrieve(SONGLENGTHS_URL, cache)
    return cache


def _songlength_cycles(sid_path, subtune):
    """Cycle budget for ``subtune`` (1-based) from the HVSC Songlengths.md5."""
    with open(sid_path, "rb") as handle:
        md5 = hashlib.md5(handle.read()).hexdigest().lower()
    line = None
    with open(_songlengths_path(), encoding="utf-8") as handle:
        for entry in handle:
            if entry.lower().startswith(md5):
                line = entry
                break
    if line is None:
        raise RuntimeError(f"no Songlengths entry for {os.path.basename(sid_path)}")
    fields = line.strip().split("=")[1].split(" ")
    field = fields[subtune - 1]
    seconds = 0.0
    if "." in field:
        field, millis = field.split(".")
        seconds += float(millis) / 1e3
    parts = [int(part) for part in field.split(":")]
    seconds += parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0]
    return int(PAL_PHI * seconds)


def _render_dump(sid_path, subtune, out_path):
    """Render ``subtune`` of ``sid_path`` to ``out_path`` via the container."""
    limit = _songlength_cycles(sid_path, subtune)
    with tempfile.TemporaryDirectory() as work:
        sid_name = os.path.basename(sid_path)
        shutil.copy(sid_path, os.path.join(work, sid_name))
        cli = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{work}:/work",
            IMAGE,
            "/usr/local/bin/vsiddump.py",
            "--dumpdir",
            "/work",
            "--sid",
            f"/work/{sid_name}",
            "-tune",
            str(subtune),
            "-limitcycles",
            str(limit),
        ]
        subprocess.run(cli, check=True, capture_output=True)
        produced = os.path.join(work, f"{_base(sid_path)}.None.dump.parquet")
        if not os.path.exists(produced):
            raise RuntimeError(f"headlessvice produced no dump for {sid_name}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.move(produced, out_path)


def _resolve_sid(hvsc_rel, sid_url):
    local = os.path.join(HVSC, hvsc_rel)
    if os.path.exists(local):
        return local
    cache = os.path.join(FIXTURE_DIR, os.path.basename(hvsc_rel))
    if not os.path.exists(cache):
        os.makedirs(FIXTURE_DIR, exist_ok=True)
        urllib.request.urlretrieve(sid_url, cache)
    return cache


def _resolve_dump(sid_path, hvsc_rel, subtune):
    base = _base(hvsc_rel)
    name = f"{base}.{subtune}.dump.parquet"
    local = os.path.join(HVSC, os.path.dirname(hvsc_rel), name)
    if os.path.exists(local):
        return local
    cache = os.path.join(FIXTURE_DIR, name)
    if not os.path.exists(cache):
        _render_dump(sid_path, subtune, cache)
    return cache


def acquire(hvsc_rel, sid_url, subtune):
    """Return (sid_path, dump_path) for one HVSC tune+subtune, building if absent.

    ``hvsc_rel`` is the path under the HVSC ``C64Music`` root (which also names the
    cache files); ``sid_url`` is the download fallback; ``subtune`` is 1-based.
    """
    sid = _resolve_sid(hvsc_rel, sid_url)
    dump = _resolve_dump(sid, hvsc_rel, subtune)
    return sid, dump


_HVSC_BASE = "https://hvsc.brona.dk/HVSC/C64Music"
# The tunes the gate tests need; rendered into FIXTURE_DIR by render_fixtures().
GATE_FIXTURES = (
    ("MUSICIANS/H/Hubbard_Rob/Monty_on_the_Run.sid", 1),
    ("MUSICIANS/H/Hubbard_Rob/5_Title_Tunes.sid", 2),
    ("MUSICIANS/J/Jammer/Grid_Runner.sid", 1),
    # GoatTracker boot-frame alignment + recover/render regressions:
    # Need_More_NOPs -- deep-offset boot frame (dump starts ~36 frames into
    # playback, past the old 32-frame window); now byte-exact via the widened,
    # window-based alignment.
    ("MUSICIANS/F/Fegolhuzz/Need_More_NOPs.sid", 1),
    # Not_Even_Human -- note bytes below FIRSTNOTE (no clean freq-table pitch);
    # the raw-note escape keeps measure from feeding log2(<=0) into the grid.
    ("MUSICIANS/C/Crowley_Owen/Not_Even_Human.sid", 1),
    # FamiCommodore -- a recovered table pointer overruns at render; must fail
    # cleanly with a descriptive RuntimeError, not a bare pygoattracker IndexError.
    ("DEMOS/A-F/FamiCommodore.sid", 1),
    # Twilight -- packed freq-table overrun: the packed player's UNPADDED freq
    # table (freqtbllo[fn..ln] | freqtblhi[fn..ln] | songtbl, L=80, firstnote=16)
    # is indexed past its end by a wavetable relative-note step, so out-of-range
    # notes read adjacent image bytes. Byte-exact only with the packed-image
    # freq table (vs the editor's zero-padded table that returns freq 0).
    ("MUSICIANS/N/No-XS/Twilight.sid", 1),
    ("MUSICIANS/L/Lft/A_Mind_Is_Born.sid", 1),  # lft algorithmic RSID (white-box)
    # multispeed (~3x): Galway's play routine fires several times per raster
    # frame -- single-CPF framing drops >50% of register changes; used to prove
    # the cadence detector + sub-frame framing are lossless.
    ("MUSICIANS/G/Galway_Martin/Times_of_Lore.sid", 1),
)


def render_fixtures():
    """Ensure every gate (.sid, .dump) is present in FIXTURE_DIR, building it (download
    + headlessvice render) if absent. Used by CI/build.sh to populate the directory
    that is then MOUNTED into the test container (the container cannot run Docker)."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    for hvsc_rel, subtune in GATE_FIXTURES:
        sid, dump = acquire(hvsc_rel, f"{_HVSC_BASE}/{hvsc_rel}", subtune)
        for src in (sid, dump):
            dst = os.path.join(FIXTURE_DIR, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy(src, dst)


if __name__ == "__main__":
    render_fixtures()
