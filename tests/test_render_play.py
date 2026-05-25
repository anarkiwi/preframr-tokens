# pylint: disable=no-member
"""Unit tests for ``preframr_tokens.render_play``."""

import argparse
import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import preframr_tokens.render_play as render_play
from preframr_tokens.reglogparser import DUMP_SUFFIX
from preframr_tokens.stfconstants import FRAME_REG


class TestLoadDf(unittest.TestCase):
    def test_dump_path_parses_and_returns_first_rotation(self):
        args = argparse.Namespace(reglog=f"/fake/song{DUMP_SUFFIX}")
        fake_df = pd.DataFrame({"reg": [FRAME_REG], "diff": [19656], "val": [0]})
        with mock.patch.object(render_play, "RegLogParser") as MockParser:
            MockParser.return_value.parse.return_value = iter([fake_df])
            out = render_play._load_df(args, logger=mock.MagicMock())
        MockParser.assert_called_once_with(args, logger=mock.ANY)
        MockParser.return_value.parse.assert_called_once_with(
            args.reglog, max_perm=1, require_pq=False, reparse=True
        )
        self.assertTrue(out.equals(fake_df))

    def test_dump_path_no_rotations_raises(self):
        args = argparse.Namespace(reglog=f"/fake/song{DUMP_SUFFIX}")
        with mock.patch.object(render_play, "RegLogParser") as MockParser:
            MockParser.return_value.parse.return_value = iter([])
            with self.assertRaises(SystemExit) as ctx:
                render_play._load_df(args, logger=mock.MagicMock())
        self.assertIn("no rotations parsed", str(ctx.exception))

    def test_parsed_path_reads_parquet_and_attaches_palettes(self):
        args = argparse.Namespace(reglog="/fake/song.0.parquet")
        fake_df = pd.DataFrame({"reg": [0], "val": [0]})
        with (
            mock.patch.object(render_play.pd, "read_parquet", return_value=fake_df),
            mock.patch.object(
                render_play,
                "load_palettes_attrs",
                return_value={"engine_fp_cluster": 3},
            ),
        ):
            out = render_play._load_df(args, logger=mock.MagicMock())
        self.assertEqual(out.attrs.get("engine_fp_cluster"), 3)


class TestMainArgparseGuards(unittest.TestCase):
    def _run_main(self, argv):
        with mock.patch.object(render_play.sys, "argv", ["render_play", *argv]):
            render_play.main()

    def test_list_audio_runs_aplay_l_and_returns(self):
        with (
            mock.patch("subprocess.run") as run,
            mock.patch.object(render_play, "_load_df") as load_df,
        ):
            self._run_main(["--list-audio"])
        run.assert_called_once_with(["aplay", "-l"], check=False)
        load_df.assert_not_called()

    def test_no_reglog_raises_system_exit(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run_main([])
        self.assertIn("reglog REQUIRED", str(ctx.exception))

    def test_missing_reglog_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does_not_exist.parquet")
            with self.assertRaises(SystemExit) as ctx:
                self._run_main([missing])
        self.assertIn("file not found", str(ctx.exception))


class TestMainPlaybackDispatch(unittest.TestCase):
    """Three playback paths: --via-wav, --prerender, default (aplay).
    Each branch needs a positive test that mocks all three audio
    primitives and asserts only the expected one was called.
    """

    def _stub_df(self):
        return pd.DataFrame(
            {
                "reg": [FRAME_REG, 0],
                "diff": [12345, 0],
                "val": [0, 0],
            }
        )

    def _make_existing_reglog(self, tmp):
        path = os.path.join(tmp, "stub.parquet")
        Path(path).touch()
        return path

    def _common_mocks(self, stack, df_to_return):
        stack.enter_context(
            mock.patch.object(render_play, "_load_df", return_value=df_to_return)
        )
        stack.enter_context(
            mock.patch.object(
                render_play,
                "prepare_df_for_audio",
                return_value=(df_to_return, {1: 1}),
            )
        )
        stack.enter_context(mock.patch.object(render_play, "LiveAnimator"))
        wav = stack.enter_context(
            mock.patch.object(render_play, "play_via_wav", return_value=100)
        )
        pre = stack.enter_context(
            mock.patch.object(
                render_play, "play_via_aplay_prerendered", return_value=200
            )
        )
        live = stack.enter_context(
            mock.patch.object(render_play, "play_via_aplay", return_value=300)
        )
        return wav, pre, live

    def test_default_path_calls_play_via_aplay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_existing_reglog(tmp)
            with mock.patch.object(render_play.sys, "argv", ["rp", path]):
                with contextlib.ExitStack() as stack:
                    wav, pre, live = self._common_mocks(stack, self._stub_df())
                    render_play.main()
            wav.assert_not_called()
            pre.assert_not_called()
            live.assert_called_once()
            self.assertEqual(live.call_args.kwargs["irq"], 12345)

    def test_via_wav_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_existing_reglog(tmp)
            with mock.patch.object(render_play.sys, "argv", ["rp", path, "--via-wav"]):
                with contextlib.ExitStack() as stack:
                    wav, pre, live = self._common_mocks(stack, self._stub_df())
                    render_play.main()
            wav.assert_called_once()
            pre.assert_not_called()
            live.assert_not_called()

    def test_prerender_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_existing_reglog(tmp)
            with mock.patch.object(
                render_play.sys, "argv", ["rp", path, "--prerender"]
            ):
                with contextlib.ExitStack() as stack:
                    wav, pre, live = self._common_mocks(stack, self._stub_df())
                    render_play.main()
            pre.assert_called_once()
            wav.assert_not_called()
            live.assert_not_called()

    def test_paplay_cmd_passed_to_aplay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_existing_reglog(tmp)
            with mock.patch.object(render_play.sys, "argv", ["rp", path, "--paplay"]):
                with contextlib.ExitStack() as stack:
                    _, _, live = self._common_mocks(stack, self._stub_df())
                    render_play.main()
            live.assert_called_once()
            cmd = live.call_args.kwargs["aplay_cmd"]
            self.assertIsNotNone(cmd)
            self.assertEqual(cmd[0], "paplay")
            self.assertIn("--rate=48000", cmd)

    def test_no_frame_reg_falls_back_to_pal_irq(self):
        df = pd.DataFrame({"reg": [0], "diff": [0], "val": [0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_existing_reglog(tmp)
            with mock.patch.object(render_play.sys, "argv", ["rp", path]):
                with contextlib.ExitStack() as stack:
                    _, _, live = self._common_mocks(stack, df)
                    render_play.main()
            self.assertEqual(live.call_args.kwargs["irq"], 19656)

    def test_animator_opened_and_closed_around_playback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_existing_reglog(tmp)
            with mock.patch.object(render_play.sys, "argv", ["rp", path, "--animate"]):
                with contextlib.ExitStack() as stack:
                    self._common_mocks(stack, self._stub_df())
                    render_play.main()
                    animator_cls = render_play.LiveAnimator
            animator_inst = animator_cls.return_value
            animator_inst.open.assert_called_once()
            animator_inst.close.assert_called_once()

    def test_playback_runtime_error_calls_animator_close_before_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_existing_reglog(tmp)
            with mock.patch.object(render_play.sys, "argv", ["rp", path, "--animate"]):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(
                            render_play, "_load_df", return_value=self._stub_df()
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            render_play,
                            "prepare_df_for_audio",
                            return_value=(self._stub_df(), {}),
                        )
                    )
                    stack.enter_context(mock.patch.object(render_play, "LiveAnimator"))
                    stack.enter_context(
                        mock.patch.object(
                            render_play,
                            "play_via_aplay",
                            side_effect=RuntimeError("simulated playback fail"),
                        )
                    )
                    with self.assertRaises(SystemExit) as ctx:
                        render_play.main()
                    animator_cls = render_play.LiveAnimator
            self.assertIn("playback failed", str(ctx.exception))
            animator_cls.return_value.close.assert_called()


if __name__ == "__main__":
    unittest.main()
