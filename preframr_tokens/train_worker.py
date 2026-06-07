import concurrent.futures
import re
import zstandard as zstd
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

UNK_TOKEN = "<unk>"
END_OF_WORD_SUFFIX = "</w>"


def _isolation_char_class(isolation_chars):
    """Return a regex character class that matches each char in
    ``isolation_chars`` exactly once. Contiguous runs of codepoints are
    collapsed to ``a-b`` ranges so the resulting class fits within the
    Oniguruma regex engine's "multibyte code range" limit -- the macros-
    on alphabet has ~50K PATTERN_REPLAY/OVERLAY chars and listing them
    """
    cps = sorted({ord(c) for c in isolation_chars})
    if not cps:
        return None
    parts = []
    start = prev = cps[0]
    for cp in cps[1:]:
        if cp == prev + 1:
            prev = cp
            continue
        parts.append(
            re.escape(chr(start))
            if start == prev
            else f"{re.escape(chr(start))}-{re.escape(chr(prev))}"
        )
        start = prev = cp
    parts.append(
        re.escape(chr(start))
        if start == prev
        else f"{re.escape(chr(start))}-{re.escape(chr(prev))}"
    )
    return "[" + "".join(parts) + "]"


def _build_unigram_pre_tokenizer(isolation_chars):
    """Compose the unigram pre-tokenizer."""
    pattern = _isolation_char_class(isolation_chars)
    if pattern is None:
        return pre_tokenizers.Punctuation()
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(pattern), behavior="isolated", invert=False
            ),
            pre_tokenizers.Punctuation(),
        ]
    )


def get_tk(tkvocab, tokenizer="bpe", initial_alphabet=None, isolation_chars=""):
    if initial_alphabet is None:
        initial_alphabet = []
    if tokenizer == "unigram":
        tk = Tokenizer(models.Unigram())
        tk.pre_tokenizer = _build_unigram_pre_tokenizer(isolation_chars)
        tk.decoder = decoders.Metaspace(replacement=" ")
        tk.normalizer = None
        trainer = trainers.UnigramTrainer(
            vocab_size=tkvocab,
            show_progress=True,
            special_tokens=[UNK_TOKEN],
            initial_alphabet=initial_alphabet,
            unk_token=UNK_TOKEN,
            shrinking_factor=0.95,
        )
        return tk, trainer
    if tokenizer == "bpe":
        tk = Tokenizer(
            models.BPE(
                dropout=None,
                unk_token=UNK_TOKEN,
                end_of_word_suffix=END_OF_WORD_SUFFIX,
                fuse_unk=False,
                byte_fallback=False,
                ignore_merges=False,
                vocab={},
                merges=[],
            )
        )
        tk.normalizer = None
        tk.pre_tokenizer = pre_tokenizers.Punctuation()
        tk.decoder = decoders.BPEDecoder()
        trainer = trainers.BpeTrainer(
            vocab_size=tkvocab,
            min_frequency=2,
            special_tokens=[UNK_TOKEN],
            limit_alphabet=tkvocab,
            initial_alphabet=initial_alphabet,
            show_progress=True,
        )
        return tk, trainer
    raise ValueError


def train_worker(
    tokenizer,
    tkvocab,
    args_tkmodel,
    uni_files,
    initial_alphabet=None,
    isolation_chars="",
):
    if initial_alphabet is None:
        initial_alphabet = []

    def read_uni(uni_file):
        with zstd.open(uni_file, "r") as f:
            return f.read()

    def reader():
        """Yield one sequence per ``.uni`` file (a bounded per-tune/block chunk) rather than a single
        giant concatenation of the whole corpus. The big single sequence stresses the unigram trainer's
        per-item pre-tokenisation/feed (the ~50K-symbol isolation alphabet + a multi-million-char string)
        and has been linked to non-deterministic SIGSEGVs; per-file chunks bound the per-item work.
        ``map`` preserves input order (the old ``as_completed`` join was order-nondeterministic, so the
        trained vocab was not reproducible) while keeping the reads parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
            for part in p.map(read_uni, uni_files):
                yield part

    tkmodel, trainer = get_tk(
        tkvocab,
        tokenizer=tokenizer,
        initial_alphabet=initial_alphabet,
        isolation_chars=isolation_chars,
    )
    if trainer is not None:
        tkmodel.train_from_iterator(reader(), trainer=trainer)
    else:
        tkmodel.train_from_iterator(reader(), vocab_size=tkvocab, show_progress=True)
    tkmodel.save(args_tkmodel)
