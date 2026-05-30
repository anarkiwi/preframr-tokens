"""Driver-truth coverage (#11): RESID share is the encoding-completeness metric. A
synthetic generator per mechanism emits the register stream that driver produces, driven
through the FULL parser, and asserts the parse classifies the correct ORN primitive with
ZERO RESID -- a RESID leak on a known mechanism is a failing completeness test (fix the
encoding, never the threshold). Real-tune RESID gaps are xfail-strict + characterized.
"""

import os
import tempfile
import unittest

import pytest

from tests.parse_probes import (
    DumpBuilder,
    cents_to_fn,
    inline_note_signature,
    inline_orn_notes,
    parse_args,
    resid_breakdown,
    skeleton_orn_summary,
    write_dump,
)
from tests.sid_fixtures import FixtureUnavailable, cache_dir, ensure_driver_fixture

from preframr_tokens.macros.skeleton_pass import LUT
from preframr_tokens.stfconstants import (
    ORN_TYPE_ARP,
    ORN_TYPE_OCTAVE,
    ORN_TYPE_PLAIN,
    ORN_TYPE_RESID,
    ORN_TYPE_SLIDE,
    ORN_TYPE_VIB,
)

RESID_MAX = 0.10
_NO_CACHE_MSG = (
    "PREFRAMR_SID_FIXTURE_CACHE unset and no local HVSC tree; real-tune layer runs "
    "only where the fixture cache/headlessvice is available"
)


def _skeleton_args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def _mechanism_note(per_frame_fns):
    """Wrap one mechanism note between plain lead-in/lead-out notes (so the parser
    keeps it and the level-change detector has clean boundaries) and parse it once."""
    builder = DumpBuilder().adsr().pw(0x800)
    builder.note([LUT[60]] * 5)
    builder.note(per_frame_fns)
    builder.note([LUT[48]] * 5)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_dump(builder, os.path.join(tmp, "mechanism.dump.parquet"))
        return list(inline_orn_notes(path, _skeleton_args()))


def _type_of(notes, expected_type):
    """The (single) note classified ``expected_type``; its offset params. Asserts the
    mechanism note was found and that NO note in the stream escaped to RESID."""
    typed = [params for orn_type, params in notes if orn_type == expected_type]
    resid = [params for orn_type, params in notes if orn_type == ORN_TYPE_RESID]
    return typed, resid


def _fixture_cache_present():
    if os.environ.get("PREFRAMR_SID_FIXTURE_CACHE"):
        return True
    return (cache_dir() / "hvsc").exists() or os.path.isdir("/scratch/preframr/hvsc")


class TestSyntheticDriverMechanisms(unittest.TestCase):
    """One generator per pitch driver mechanism; each asserts the correct primitive and
    zero RESID (the synthetic core, copyright-free, always runs)."""

    def test_octave_arp(self):
        """Hubbard octave arp (note / note+12 @50Hz) -> ORN_TYPE_OCTAVE, no RESID."""
        notes = _mechanism_note(
            [LUT[60 + (0 if f % 2 == 0 else 12)] for f in range(12)]
        )
        typed, resid = _type_of(notes, ORN_TYPE_OCTAVE)
        self.assertTrue(typed, notes)
        self.assertEqual(resid, [], notes)
        self.assertTrue(set(typed[0]) <= {0, 12}, typed)

    def test_table_arp(self):
        """Tracker table arp (note-relative [0,+4,+7] cycle) -> ORN_TYPE_ARP with the
        correct period, no RESID."""
        notes = _mechanism_note([LUT[60 + [0, 4, 7][f % 3]] for f in range(18)])
        typed, resid = _type_of(notes, ORN_TYPE_ARP)
        self.assertTrue(typed, notes)
        self.assertEqual(resid, [], notes)
        self.assertEqual(list(typed[0]), [0, 4, 7], typed)

    def test_vibrato(self):
        """Sub-semitone vibrato wobble -> ORN_TYPE_VIB with a non-zero depth bucket, no
        RESID (the content-tier floor drops the wobble to a learnable depth/rate)."""
        notes = _mechanism_note(
            [cents_to_fn(67, 25.0 if f % 2 == 0 else -25.0) for f in range(12)]
        )
        typed, resid = _type_of(notes, ORN_TYPE_VIB)
        self.assertTrue(typed, notes)
        self.assertEqual(resid, [], notes)
        self.assertGreater(typed[0][0], 0, typed)

    def test_slide(self):
        """Portamento ramp toward a target -> ORN_TYPE_SLIDE with target ~= the reached
        offset, no RESID."""
        notes = _mechanism_note([LUT[60 + min(7, f)] for f in range(1, 16)])
        typed, resid = _type_of(notes, ORN_TYPE_SLIDE)
        self.assertTrue(typed, notes)
        self.assertEqual(resid, [], notes)
        self.assertGreaterEqual(abs(typed[0][0]), 5, typed)

    def test_plain_held(self):
        """A plain held note -> ORN_TYPE_PLAIN, no RESID."""
        notes = _mechanism_note([LUT[64]] * 10)
        typed, resid = _type_of(notes, ORN_TYPE_PLAIN)
        self.assertTrue(typed, notes)
        self.assertEqual(resid, [], notes)

    def test_resid_is_a_completeness_signal(self):
        """A genuinely aperiodic, wide, non-monotone pitch jumble has no driver
        primitive and SHOULD escape to RESID -- documenting that RESID is the
        completeness metric: it fires exactly when no mechanism models the note."""
        jumble = [LUT[60 + o] for o in (0, 11, 3, 17, 1, 14, 6, 20, 2, 9)]
        notes = _mechanism_note(jumble)
        resid = [p for t, p in notes if t == ORN_TYPE_RESID]
        self.assertTrue(resid, notes)


class TestRealDriverEncoding(unittest.TestCase):
    """Per-driver real-tune fixtures (cached, never committed, regenerate-or-fail). A
    real regression guard: the skeleton+ornament encoding fires on actual driver output
    of every known driver family. The per-tune RESID-share gate is the xfail layer below
    (the threshold is not principled on full tunes; see the characterization)."""

    @classmethod
    def setUpClass(cls):
        if not _fixture_cache_present():
            raise unittest.SkipTest(_NO_CACHE_MSG)

    def _summary(self, name):
        try:
            dump = str(ensure_driver_fixture(name))
        except FixtureUnavailable as err:
            raise AssertionError(
                f"fixture cache present but {name} dump unavailable: {err}"
            ) from err
        return skeleton_orn_summary(dump, _skeleton_args())

    def test_commando_encodes(self):
        """Commando (Hubbard octave-arp + portamento family) parses to a skeleton."""
        summary = self._summary("commando")
        self.assertGreater(summary["skel"], 0, summary)
        self.assertGreater(summary["orn"], 0, summary)

    def test_camerock_encodes(self):
        """Camerock (DRAX) parses to a skeleton."""
        summary = self._summary("camerock")
        self.assertGreater(summary["skel"], 0, summary)
        self.assertGreater(summary["orn"], 0, summary)


class TestRealDriverResidGap(unittest.TestCase):
    """The known RESID completeness gap on real tunes, xfail-strict so the suite is GREEN
    while the gap is explicit and flips to XPASS (alerting) when the missing mechanism is
    later modelled. Characterized: the leak is dominated by fast-melodic-run
    under-segmentation (recoverable as notes), not legit glissando."""

    @classmethod
    def setUpClass(cls):
        if not _fixture_cache_present():
            raise unittest.SkipTest(_NO_CACHE_MSG)

    def _resid_note_share(self, name):
        try:
            dump = str(ensure_driver_fixture(name))
        except FixtureUnavailable as err:
            raise AssertionError(
                f"fixture cache present but {name} dump unavailable: {err}"
            ) from err
        summary = skeleton_orn_summary(dump, _skeleton_args())
        self.assertGreater(summary["orn"], 0, summary)
        return summary["resid"] / summary["orn"], summary

    def test_trap_resid_gap(self):
        """Trap/Daglish: CLOSED by the fast-melodic-run de-merge (#13) -- 0.44 -> 0.01 note-share,
        the under-segmented run folded into SKEL notes; residual is a thin aperiodic tail.
        """
        share, summary = self._resid_note_share("trap")
        self.assertLessEqual(share, RESID_MAX, summary)

    @pytest.mark.xfail(
        strict=True,
        reason="RESID gap (Baggis/Goto80): the fast-melodic-run portion is CLOSED by #13 "
        "(0.66 -> 0.26 note-share); the remainder is WIDE APERIODIC content (span 51-71 semitones, "
        "<=8 distinct -- octave-jump wavetable effects / noise), a DISTINCT primitive, not the "
        "fast-run mechanism. Tracking separately; do NOT raise RESID_MAX or widen the run-split "
        "(would forge spurious giant-interval notes)",
    )
    def test_baggis_resid_gap(self):
        share, summary = self._resid_note_share("baggis")
        self.assertLessEqual(share, RESID_MAX, summary)


class TestResidCharacterization(unittest.TestCase):
    """Regression guard for the fast-melodic-run de-merge (#13). Pre-fix, the Trap/Baggis RESID
    was DOMINATED by the fast-melodic-run bucket (Trap ~0.93, Baggis ~0.76 of RESID frames) --
    the dominant shared under-segmentation. The fix folds those runs into SKEL notes, so the
    fast-melodic-run frame-fraction must now be SMALL; reverting the fix sends it back up and
    fails this test. The residual leak is the wide/aperiodic primitive (a separate gap).
    """

    FAST_RUN_FRAC_MAX = 0.05

    @classmethod
    def setUpClass(cls):
        if not _fixture_cache_present():
            raise unittest.SkipTest(_NO_CACHE_MSG)

    def test_fast_run_gap_closed(self):
        for name in ("trap", "baggis"):
            try:
                dump = str(ensure_driver_fixture(name))
            except FixtureUnavailable as err:
                raise AssertionError(
                    f"fixture cache present but {name} dump unavailable: {err}"
                ) from err
            breakdown = resid_breakdown(dump, _skeleton_args())
            by_frame = breakdown["by_frame"]
            total = sum(by_frame.values())
            fast_run_frac = by_frame.get("fast-melodic-run", 0) / max(total, 1)
            self.assertLessEqual(
                fast_run_frac, self.FAST_RUN_FRAC_MAX, (name, breakdown)
            )


class TestProvenanceInvariance(unittest.TestCase):
    """P7 / universal driver (#11.4): the SAME musical gesture must encode to the SAME tokens
    regardless of register-level provenance. The deterministic guarantee that the encoder is
    provenance-agnostic (no per-driver / per-pitch branching): ORN is transposition- and
    duration-invariant; the content-tier semitone floor is invariant to driver tuning.
    """

    def _sig(self, notes):
        builder = DumpBuilder().adsr().pw(0x800)
        for per_frame_fns in notes:
            builder.note(per_frame_fns)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_dump(builder, os.path.join(tmp, "prov.dump.parquet"))
            return inline_note_signature(path, _skeleton_args())

    @staticmethod
    def _orn_seq(sig):
        return [(orn_type, offs) for _, orn_type, offs in sig]

    def test_transposition_invariant_ornament(self):
        """An arp `[0,+4,+7]` encodes to the SAME ORN whether based at C4 or C5 -- the ornament
        is note-relative, so transposing the whole phrase leaves the ORN token stream unchanged.
        """

        def phrase(base):
            arp = [LUT[base + [0, 4, 7][f % 3]] for f in range(18)]
            return self._sig([[LUT[base - 5]] * 5, arp, [LUT[base - 12]] * 5])

        low, high = phrase(60), phrase(72)
        self.assertIn(ORN_TYPE_ARP, [t for t, _ in self._orn_seq(low)], low)
        self.assertEqual(self._orn_seq(low), self._orn_seq(high))

    def test_tuning_invariant_skeleton(self):
        """A melody at exact tuning vs a constant sub-semitone detune (a different driver's
        tuning) encodes to the IDENTICAL SKEL+ORN tokens -- the content-tier floor absorbs the
        cents, so provenance tuning does not change the melody the model sees."""
        mel = (60, 64, 67)
        exact = self._sig([[LUT[n]] * 6 for n in mel])
        detuned = self._sig([[cents_to_fn(n, 15.0)] * 6 for n in mel])
        self.assertTrue(exact)
        self.assertEqual(exact, detuned)

    def test_duration_invariant_ornament(self):
        """The same arp cycle over 12 vs 18 frames encodes to the SAME (constant-size) ORN --
        ornament tokens describe the cycle, not its length, so note duration is not provenance.
        """

        def arp_phrase(frames):
            arp = [LUT[60 + [0, 4, 7][f % 3]] for f in range(frames)]
            return self._orn_seq(self._sig([[LUT[55]] * 5, arp, [LUT[48]] * 5]))

        short, long = arp_phrase(12), arp_phrase(18)
        self.assertIn(ORN_TYPE_ARP, [t for t, _ in short], short)
        self.assertEqual(short, long)


if __name__ == "__main__":
    unittest.main()
