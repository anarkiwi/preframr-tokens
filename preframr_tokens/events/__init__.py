"""The v3 event token model: a musical event stream distinct from the register-write df. The df is only
the byte-exact ground-truth oracle (:mod:`preframr_tokens.events.oracle`); the stream codec
(:mod:`preframr_tokens.events.stream`) encodes it to canonical event tokens and decodes them back to the
exact ordered write stream. Every field is a complete value over a small fixed alphabet (no ids, no
escape), so BPE over the token stream is the corpus-global dictionary."""
