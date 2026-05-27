"""Mine a ``MotifDict`` from a dump corpus: parse each dump with the parser/macro
args training uses (motif pass forced OFF so it sees un-collapsed atoms), group the
per-block atom streams by composer (the dump's parent dir), and run the
cross-composer greedy miner. Torch-free, so it runs in the parser environment; the
framework mine CLI and the experiment pre_run_hook both drive it."""

import logging
from pathlib import Path

from preframr_tokens.blocks import glob_dumps, iter_voiced_blocks
from preframr_tokens.macros.motif_pass import mine_motifs, mine_templates, _atoms_of
from preframr_tokens.reglogparser import RegLogParser

__all__ = ["mine_dict_from_dumps"]


def _composer(dump_path):
    """Composer label for cross-composer weighting: the dump's parent dir name
    (matches the runner's ``<composer>/<basename>`` staging layout)."""
    return Path(dump_path).parent.name or "_root_"


def mine_dict_from_dumps(
    args,
    reglogs,
    max_files=0,
    k=256,
    min_count=3,
    min_composers=3,
    version=1,
    logger=logging,
):
    """Parse ``reglogs`` (a comma-separated dump-glob spec) with ``args`` and mine a
    ``MotifDict``. ``motif_pass`` is forced off and mining runs over the SAME
    self-contained voiced blocks the encode path collapses (``iter_voiced_blocks``),
    so the mined merges match at encode time. Each block is one stream tagged by
    composer; only the identity rotation is used (matching ``--max-perm 1``)."""
    args.motif_pass = False
    parser = RegLogParser(args, logger)
    block_parser = RegLogParser(args, logger)
    stride = getattr(args, "block_stride", None)
    seq_len = getattr(args, "seq_len", 4096)
    streams = []
    composers = []
    files = glob_dumps(reglogs, max_files, require_pq=False)
    for name in files:
        composer = _composer(name)
        try:
            for df in parser.parse(name, max_perm=1, require_pq=False, reparse=True):
                for voiced in iter_voiced_blocks(
                    df, seq_len, block_parser, {}, stride=stride
                ):
                    if voiced.empty:
                        continue
                    streams.append(_atoms_of(voiced))
                    composers.append(composer)
        except (AssertionError, ValueError, KeyError) as exc:
            logger.warning("motif mining: dropping %s: %s", name, exc)
    if not streams:
        raise ValueError(f"motif mining: no parseable dumps matched {reglogs!r}")
    logger.info(
        "motif mining: %u blocks across %u composers; k=%u min_count=%u min_composers=%u",
        len(streams),
        len(set(composers)),
        k,
        min_count,
        min_composers,
    )
    miner = mine_templates if version == 2 else mine_motifs
    return miner(
        streams, composers, k=k, min_count=min_count, min_composers=min_composers
    )
