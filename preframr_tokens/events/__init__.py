"""The inline-event token codec (the white-box decompiler representation): a SID
dump's settled per-frame state becomes one time-ordered stream of relative-pitch and
delta-run gestures over a fixed small atom alphabet (no ids, no escape, no frozen
table). :mod:`preframr_tokens.events.oracle` settles the byte-exact write stream;
:mod:`preframr_tokens.events.stream` round-trips it; BPE over the atoms is the dictionary.
"""
