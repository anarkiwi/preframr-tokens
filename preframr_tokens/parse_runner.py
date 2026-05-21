"""Parallel dump-parsing orchestrator: glob dumps, parse via RegLogParser, persist per-rotation .parquet rotations. Pure torch-free + stdlib + preframr_tokens internals; main-repo `preframr/parse.py` is a thin argparse shim that calls into here."""

import concurrent.futures
import multiprocessing
import traceback

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from preframr_tokens.blocks import glob_dumps
from preframr_tokens.reglog_helpers import dump_palettes_attrs, tighten_persist_dtypes
from preframr_tokens.reglogparser import RegLogParser


def write_df(args, logger, name):
    """Parse one dump file via RegLogParser; for each rotation, dump the dataframe to a .{i}.parquet sidecar (with palette attrs + dtype tightening)."""
    log_parser = RegLogParser(args, logger)
    base_name = name.replace(".dump.parquet", "")
    try:
        for i, df in enumerate(
            log_parser.parse(name, max_perm=99, require_pq=False, reparse=True)
        ):
            pq_name = base_name + f".{i}.parquet"
            dump_palettes_attrs(df.attrs, pq_name)
            df.attrs = {}
            tighten_persist_dtypes(df)
            df.to_parquet(pq_name, engine="pyarrow", compression="zstd")
    except Exception as e:  # pylint: disable=broad-except
        traceback_str = "".join(traceback.format_tb(e.__traceback__))
        raise ValueError(f"{name}: {traceback_str}") from e


def parse_corpus(args, logger):
    """Top-level parallel parse orchestrator. Globs dump_files per args.reglogs / args.max_files (+ optional dump_meta filters), parses each in a ProcessPoolExecutor worker, writes .{i}.parquet per rotation. Returns nothing; on disk artefacts are the deliverable."""
    irq_lo = getattr(args, "meta_irq_lo", 0)
    irq_hi = getattr(args, "meta_irq_hi", 0)
    irq_range = (int(irq_lo), int(irq_hi)) if (irq_lo or irq_hi) else None
    with logging_redirect_tqdm():
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count()
        ) as executor:
            futures = []
            for name in glob_dumps(
                args.reglogs,
                args.max_files,
                require_pq=False,
                exclude_digi=getattr(args, "meta_exclude_digi", False),
                irq_range=irq_range,
                require_meta=getattr(args, "meta_require", False),
            ):
                futures.append(executor.submit(write_df, args, logger, name))
            with tqdm(total=len(futures)) as t:
                for future in concurrent.futures.as_completed(futures):
                    assert not future.exception(), future.exception()
                    t.update(1)
