"""Merge-table boundary audit for a trained unigram over the event alphabet: per vocab piece, how many
atoms it welds, whether it crosses a VOICE/KEYFRAME boundary, how many event kinds it spans, and whether
it is all varint digits. DT digits share the varint range with value digits, so cross-FRAME welds through
a DT cannot be isolated without killing value merges -- this audit measures them instead. Run it after any
unigram train to spot re-multiplexing merges the alphabet meant to keep apart."""

from __future__ import annotations

import pandas as pd

from preframr_tokens.events import stream

_AUDIT_COLUMNS = ["piece_id", "n_atoms", "crosses_voice", "n_kinds", "all_digits"]
_VOICE_KF = set(range(stream.VOICE_BASE, stream.VOICE_BASE + 4)) | {stream.KEYFRAME}


def _piece_atoms(tokenizer, piece):
    """Decode one vocab piece string to its atom ids (n-space minus one)."""
    return [int(i) - 1 for i in tokenizer.decode_unicode(piece)]


def audit_vocab(tokenizer) -> pd.DataFrame:
    """Per-piece merge audit of a RegTokenizer trained over ``events_alphabet()``: one row per vocab
    piece (skipping ``<unk>``) flagging voice/KEYFRAME-crossing welds, the event-kind span, and
    all-digit pieces."""
    rows = []
    for piece, piece_id in tokenizer.tkmodel.get_vocab().items():
        if piece == "<unk>":
            continue
        atoms = _piece_atoms(tokenizer, piece)
        n_atoms = len(atoms)
        rows.append(
            {
                "piece_id": int(piece_id),
                "n_atoms": n_atoms,
                "crosses_voice": n_atoms > 1 and any(a in _VOICE_KF for a in atoms),
                "n_kinds": sum(1 for a in atoms if stream.TUNING <= a <= stream.G_RAMP),
                "all_digits": all(
                    stream.VAR_BASE <= a < stream.REG_BASE for a in atoms
                ),
            }
        )
    return pd.DataFrame(rows, columns=_AUDIT_COLUMNS)


def summarize(frame) -> dict:
    """Corpus-level merge summary over the multi-atom pieces of an :func:`audit_vocab` frame."""
    multi = frame[frame["n_atoms"] > 1]
    n_multi = int(len(multi))
    n_crossing = int(multi["crosses_voice"].sum())
    return {
        "n_pieces": int(len(frame)),
        "n_multi_atom": n_multi,
        "n_crossing_voice": n_crossing,
        "frac_crossing_voice": (n_crossing / n_multi) if n_multi else 0.0,
        "n_multi_kind": int((multi["n_kinds"] > 1).sum()),
    }


if __name__ == "__main__":
    import sys

    from preframr_tokens.events import dataset as _dataset
    from preframr_tokens.regtokenizer import RegTokenizer

    if len(sys.argv) != 2:
        print(
            "usage: python -m preframr_tokens.bpe_audit <tkmodel.json>", file=sys.stderr
        )
        sys.exit(2)
    _tok = RegTokenizer(None, _dataset.events_alphabet())
    _tok._resync_splitters_from_tokens()  # pylint: disable=protected-access
    with open(sys.argv[1]) as _fh:
        _tok.load(_fh.read(), _dataset.events_alphabet())
    _frame = audit_vocab(_tok)
    print(summarize(_frame))
    _crossing = _frame[_frame["crosses_voice"]].sort_values("n_atoms", ascending=False)
    for _row in _crossing.head(20).itertuples():
        _piece = next(
            p
            for p, i in _tok.tkmodel.get_vocab().items()
            if int(i) == int(_row.piece_id)
        )
        print(int(_row.piece_id), _piece_atoms(_tok, _piece))
