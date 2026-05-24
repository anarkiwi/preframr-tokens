"""On-demand SID register-dump fixtures regenerated from HVSC, so no
copyrighted song data is committed. The fidelity tests need a real tune, so this
downloads the ``.sid`` from HVSC, renders a register dump with ``vsid`` inside
the ``anarkiwi/headlessvice`` image, and caches the ``head``/``26s`` slices under
``$PREFRAMR_SID_FIXTURE_CACHE``; :class:`FixtureUnavailable` signals the skip."""

import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "FixtureUnavailable",
    "SidDumpSpec",
    "GRID_RUNNER",
    "ensure_dumps",
    "grid_runner_dumps",
]

PAL_PHI = 985248
HVSC_BASE_URL = "http://www.hvsc.c64.org/download/C64Music"
DUMP_IMAGE = "anarkiwi/headlessvice"
DUMP_COLUMNS = ("clock", "irq", "chipno", "reg", "val")
DUMP_TIMEOUT_S = 240
DUMP_ATTEMPTS = 3
MAX_REG = 24
RAW_NAME = "raw.csv"
VSID_BIN = "/usr/local/bin/vsid"
_VSID_FLAGS = (
    "-console -logfile /dev/null +logtofile +logtostdout -debug -warp "
    "-sound -soundwarpmode 1 -sounddev dump"
)
_REDUCE_MASKS = ((3, 0x0F), (10, 0x0F), (17, 0x0F), (21, 0x07), (23, 0xFF - 0x08))
_RAW_COLUMNS = ("clock_diff", "irq_diff", "nmi_diff", "chipno", "reg", "val")


class FixtureUnavailable(RuntimeError):
    """Raised when a dump can't be built (no Docker / image / network)."""


@dataclass(frozen=True)
class SidDumpSpec:
    """A reproducible register dump from one HVSC tune: ``hvsc_path`` relative to
    ``C64Music``, 1-based ``tune``, and PAL wall-clock spans. The ``wide`` dump is
    rendered with ``-limitcycles`` and the ``head`` dump is a leading clock-bounded
    slice; ``slug`` names the cache files (``<slug>_<n>s`` / ``<slug>_head``)."""

    slug: str
    hvsc_path: str
    tune: int = 1
    wide_seconds: int = 26
    head_seconds: int = 3

    @property
    def sid_filename(self) -> str:
        return self.hvsc_path.rsplit("/", 1)[-1]

    @property
    def limitcycles(self) -> int:
        return self.wide_seconds * PAL_PHI

    @property
    def head_cutoff_cycles(self) -> int:
        return self.head_seconds * PAL_PHI

    @property
    def wide_name(self) -> str:
        return f"{self.slug}_{self.wide_seconds}s.dump.parquet"

    @property
    def head_name(self) -> str:
        return f"{self.slug}_head.dump.parquet"


GRID_RUNNER = SidDumpSpec(
    slug="grid_runner",
    hvsc_path="MUSICIANS/J/Jammer/Grid_Runner.sid",
)


def cache_dir() -> Path:
    """Directory holding cached ``.sid`` sources and rendered dumps, from
    ``$PREFRAMR_SID_FIXTURE_CACHE`` else ``$XDG_CACHE_HOME/preframr-tokens``."""
    env = os.environ.get("PREFRAMR_SID_FIXTURE_CACHE")
    if env:
        base = Path(env)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        base = Path(xdg) / "preframr-tokens" / "sid-fixtures"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_valid_dump(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        cols = pd.read_parquet(path).columns
    except Exception:  # pylint: disable=broad-except
        return False
    return list(cols) == list(DUMP_COLUMNS)


def _have_docker_image() -> bool:
    if shutil.which("docker") is None:
        return False
    inspect = subprocess.run(
        ["docker", "image", "inspect", DUMP_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspect.returncode == 0:
        return True
    pull = subprocess.run(
        ["docker", "pull", DUMP_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return pull.returncode == 0


def _download_sid(spec: SidDumpSpec, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = f"{HVSC_BASE_URL}/{spec.hvsc_path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise FixtureUnavailable(f"could not download {url}: {err}") from err
    if not data.startswith((b"PSID", b"RSID")):
        raise FixtureUnavailable(f"{url} did not return a SID file")
    dest.write_bytes(data)


def _vsid_dump_once(spec: SidDumpSpec, build: Path) -> str:
    """Render one vsid dump to ``build/raw.csv``; "" on success else a note. vsid
    writes a regular file (not the image's FIFO, which deadlocks if vsid dies at
    startup), exits non-zero on ``-limitcycles`` so success is judged by output,
    and needs its VICE state dir recreated or it segfaults."""
    name = f"sid-dump-{uuid.uuid4().hex[:12]}"
    sid = f"/scratch/preframr/{spec.sid_filename}"
    raw = f"/scratch/preframr/{RAW_NAME}"
    vsid = (
        f"mkdir -p /root/.local/state/vice && exec {VSID_BIN} {_VSID_FLAGS} "
        f"-soundarg {raw} -tune {spec.tune} -limitcycles {spec.limitcycles} {sid}"
    )
    raw_path = build / RAW_NAME
    raw_path.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "-v",
                f"{build}:/scratch/preframr",
                DUMP_IMAGE,
                "sh",
                "-c",
                vsid,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DUMP_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return f"timed out after {DUMP_TIMEOUT_S}s"
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return "vsid produced no register stream (startup crash?)"
    return ""


def _reduce_res(df: pd.DataFrame) -> pd.DataFrame:
    """Mask sub-bit-depth register bits. Mirrors vsiddump.py:reduce_res()."""
    df = df.copy()
    for reg, mask in _REDUCE_MASKS:
        m = df["reg"] == reg
        df.loc[m, "val"] = df.loc[m, "val"] & mask
    return df


def _squeeze_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop frames whose full per-chip register state is unchanged. Mirrors
    vsiddump.py:squeeze_changes()."""
    diff_cols = df["reg"].unique()
    out = []
    for _, chip_df in df.groupby("chipno"):
        reg_df = (
            chip_df.pivot(columns="reg", values="val")
            .astype(pd.UInt32Dtype())
            .ffill()
            .fillna(0)
        )
        reg_df = reg_df.loc[
            (reg_df[diff_cols].shift(fill_value=0) != reg_df[diff_cols]).any(axis=1)
        ]
        out.append(reg_df.join(chip_df)[df.columns])
    return pd.concat(out).sort_values("clock").reset_index(drop=True)


def _postprocess_raw(raw_path: Path) -> pd.DataFrame:
    """Turn vsid's raw register-write CSV into the canonical dump DataFrame,
    byte-for-byte equivalent to the image's vsiddump.py post-processing."""
    df = pd.read_csv(raw_path, sep=r"\s+", header=None, names=list(_RAW_COLUMNS))
    df["clock"] = df["clock_diff"].cumsum()
    df["irq"] = (df["clock"] - df["irq_diff"]).clip(lower=0)
    df = df[df["reg"] <= MAX_REG]
    df = df[list(DUMP_COLUMNS)]
    df = _reduce_res(df)
    df = _squeeze_changes(df)
    return df.astype(
        {
            "clock": pd.UInt32Dtype(),
            "irq": pd.UInt32Dtype(),
            "chipno": pd.UInt8Dtype(),
            "reg": pd.UInt8Dtype(),
            "val": pd.UInt8Dtype(),
        }
    )


def _render_wide_dump(spec: SidDumpSpec, sid_path: Path) -> pd.DataFrame:
    """Render the full ``wide`` dump via vsid in the headlessvice container,
    retrying an intermittent startup crash up to ``DUMP_ATTEMPTS`` times before
    giving up with :class:`FixtureUnavailable`."""
    build = Path(tempfile.mkdtemp(prefix="sid-dump-"))
    try:
        shutil.copy(sid_path, build / spec.sid_filename)
        notes = []
        for attempt in range(1, DUMP_ATTEMPTS + 1):
            note = _vsid_dump_once(spec, build)
            if not note:
                return _postprocess_raw(build / RAW_NAME)
            notes.append(f"attempt {attempt}: {note}")
        raise FixtureUnavailable(
            f"vsid produced no dump for {spec.slug} after "
            f"{DUMP_ATTEMPTS} attempts:\n" + "\n".join(notes)
        )
    finally:
        shutil.rmtree(build, ignore_errors=True)


def ensure_dumps(spec: SidDumpSpec = GRID_RUNNER) -> tuple[Path, Path]:
    """Return ``(head_path, wide_path)`` for ``spec``, building and caching them
    first; raises :class:`FixtureUnavailable` if the dump can't be produced."""
    cache = cache_dir()
    head_path = cache / spec.head_name
    wide_path = cache / spec.wide_name
    if _is_valid_dump(head_path) and _is_valid_dump(wide_path):
        return head_path, wide_path
    if not _have_docker_image():
        raise FixtureUnavailable(
            f"Docker image {DUMP_IMAGE} unavailable; cannot regenerate "
            f"{spec.slug} dump"
        )
    sid_path = cache / spec.sid_filename
    _download_sid(spec, sid_path)
    wide = _render_wide_dump(spec, sid_path)
    head = wide[wide["clock"] < spec.head_cutoff_cycles].reset_index(drop=True)
    wide.to_parquet(wide_path, compression="zstd")
    head.to_parquet(head_path, compression="zstd")
    return head_path, wide_path


def grid_runner_dumps() -> tuple[Path, Path]:
    """Convenience accessor for the Grid Runner ``(head, 26s)`` dumps."""
    return ensure_dumps(GRID_RUNNER)
