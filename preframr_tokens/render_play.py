#!/usr/bin/env python3
"""Live render: parse a SID dump file, play it through the audio
device while showing per-voice ASCII activity.
"""

import argparse
import os
import sys

import pandas as pd

from preframr_audio.audio_driver import (
    play_via_aplay,
    play_via_aplay_prerendered,
    play_via_wav,
)
from preframr_audio.live_animator import LiveAnimator
from preframr_tokens import load_palettes_attrs
from preframr_tokens import (
    DUMP_SUFFIX,
    RegLogParser,
    prepare_df_for_audio,
    read_initial_irq,
)
from preframr_audio.sidwav import sidq
from preframr_tokens.tokenizer_config import NAMED_CONFIGS, named_config
from preframr_tokens.utils import get_logger


def _load_df(args, logger):
    """Return one rotation's df for the given ``--reglog`` path."""
    path = args.reglog
    if path.endswith(DUMP_SUFFIX):
        parser = RegLogParser(args, logger=logger)
        rotations = list(parser.parse(path, max_perm=1, require_pq=False, reparse=True))
        if not rotations:
            raise SystemExit(f"no rotations parsed from {path}")
        return rotations[0]
    df = pd.read_parquet(path)
    df.attrs.update(load_palettes_attrs(path))
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Play a SID dump file live via aplay + ASCII animation."
    )
    ap.add_argument(
        "reglog",
        nargs="?",
        default=None,
        help="Path to a .dump.parquet (re-parsed) or a parsed .parquet rotation.",
    )
    ap.add_argument("--cents", type=int, default=50)
    ap.add_argument(
        "--config",
        default="full_macros",
        choices=sorted(NAMED_CONFIGS),
        help=(
            "Parser macro config used when re-parsing a .dump.parquet. "
            "Render-equivalent across configs (macros are tokenization "
            "transforms); only affects the .dump.parquet reparse path."
        ),
    )
    ap.add_argument(
        "--animate",
        action="store_true",
        help="Show per-voice ASCII oscilloscope while playing.",
    )
    ap.add_argument(
        "--audio-device",
        default=None,
        help=(
            "ALSA device name for aplay (-D flag). e.g. plughw:0,3 for "
            "HDMI on card 0, plughw:2,0 for first USB audio card. "
            "Without this, aplay picks the system default which usually "
            "fails inside docker. List options with --list-audio."
        ),
    )
    ap.add_argument(
        "--paplay",
        action="store_true",
        help=(
            "Use ``paplay`` (native PulseAudio) instead of ``aplay`` "
            "(ALSA -> pulse bridge). Avoids the alsa-pulse plugin's "
            "userspace IPC jitter; recommended on hosts where the "
            "default aplay path produces choppy playback."
        ),
    )
    ap.add_argument(
        "--prerender",
        action="store_true",
        help=(
            "Diagnostic: render the entire stream to memory first, "
            "then stream to the audio device in one shot. Removes "
            "all real-time concerns on the producer side. If "
            "playback is still uneven through this path, the "
            "bottleneck is downstream of our code (host pulse / "
            "audio device buffer settings)."
        ),
    )
    ap.add_argument(
        "--via-wav",
        action="store_true",
        help=(
            "Render to a temporary wav file, then exec aplay/paplay "
            "against the file (rather than streaming via stdin). "
            "Workaround for hosts where stdin streaming to "
            "aplay/paplay is uneven; file-mode players use larger "
            "client-side buffering. Animator (if any) ticks via a "
            "wallclock thread alongside the playback subprocess."
        ),
    )
    ap.add_argument(
        "--keep-wav",
        default=None,
        help=(
            "With --via-wav, write the intermediate wav to this path "
            "instead of a temp file (and do not delete on exit)."
        ),
    )
    ap.add_argument(
        "--list-audio",
        action="store_true",
        help="List ALSA playback devices via ``aplay -l`` and exit.",
    )
    args = ap.parse_args()
    if args.list_audio:
        import subprocess  # noqa: WPS433

        subprocess.run(["aplay", "-l"], check=False)
        return
    if not args.reglog:
        raise SystemExit("reglog REQUIRED (path to .dump.parquet or .parquet)")
    if not os.path.exists(args.reglog):
        raise SystemExit(f"file not found: {args.reglog}")

    for _k, _v in vars(named_config(args.config, cents=args.cents)).items():
        if not hasattr(args, _k):
            setattr(args, _k, _v)

    logger = get_logger("INFO")
    df = _load_df(args, logger)

    irq = read_initial_irq(df)
    logger.info("loaded %d rows, irq=%d cycles", len(df), irq)

    df, reg_widths = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    logger.info("audio-ready df: %d rows", len(df))

    use_live_keys = args.animate and not (args.via_wav or args.prerender)
    animator = (
        LiveAnimator(
            title=os.path.basename(args.reglog),
            poll_keys=use_live_keys,
        )
        if args.animate
        else None
    )
    if animator is not None:
        animator.open()
    paplay_cmd = None
    if args.paplay:
        paplay_cmd = [
            "paplay",
            "--rate=48000",
            "--channels=1",
            "--format=s16le",
            "--raw",
            "--latency-msec=200",
        ]
    try:
        if args.via_wav:
            n = play_via_wav(
                df,
                reg_widths=reg_widths,
                irq=irq,
                cents=args.cents,
                audio_device=args.audio_device,
                use_paplay=args.paplay,
                keep_wav=args.keep_wav,
                on_frame=(animator.on_frame if animator else None),
            )
        elif args.prerender:
            n = play_via_aplay_prerendered(
                df,
                reg_widths=reg_widths,
                irq=irq,
                cents=args.cents,
                audio_device=args.audio_device,
                aplay_cmd=paplay_cmd,
                on_frame=(animator.on_frame if animator else None),
            )
        else:
            n = play_via_aplay(
                df,
                reg_widths=reg_widths,
                irq=irq,
                cents=args.cents,
                audio_device=args.audio_device,
                aplay_cmd=paplay_cmd,
                on_frame=(animator.on_frame if animator else None),
                mute_voices=(animator.muted_voices if animator else None),
                should_quit=(animator.is_quit if animator else None),
            )
    except RuntimeError as e:
        if animator is not None:
            animator.close()
            animator = None
        sys.exit(f"playback failed: {e}")
    finally:
        if animator is not None:
            animator.close()
    print(f"played {n} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
