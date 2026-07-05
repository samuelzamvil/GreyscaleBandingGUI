"""BurnBands command-line front end. All logic lives in burnbands.core."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, core


def _parse_breakpoints(text: str) -> list[float]:
    try:
        return [float(p) for p in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Breakpoints must be comma-separated numbers, got {text!r}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="burnbands",
        description=(
            "Split a grayscale image into tonal band PNGs for laser "
            "engraving (LightBurn-ready, DPI-stamped, transparent by "
            "default)."
        ),
    )
    parser.add_argument("input", type=Path, help="Input image (color or grayscale)")
    parser.add_argument(
        "-o", "--out-dir", type=Path, required=True, help="Output directory"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b", "--bands", type=int, default=4, help="Number of even bands (default 4)"
    )
    group.add_argument(
        "-p",
        "--breakpoints",
        type=_parse_breakpoints,
        help="Custom percentage breakpoints, e.g. 0,20,45,70,100",
    )
    parser.add_argument(
        "--invert", action="store_true", help="Invert luminance before banding"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI stamped into PNGs (default 300)"
    )
    parser.add_argument(
        "--white-bg",
        action="store_true",
        help="Opaque white background instead of transparency",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    percentages = (
        args.breakpoints
        if args.breakpoints is not None
        else core.even_breakpoints(args.bands)
    )

    try:
        gray, valid = core.load_grayscale(args.input, invert=args.invert)
        manifest = core.export_bands(
            gray,
            percentages,
            args.out_dir,
            dpi=args.dpi,
            white_bg=args.white_bg,
            source_name=args.input.name,
            invert=args.invert,
            valid=valid,
        )
    except core.BandingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(manifest.bands)} bands to {args.out_dir}")
    print(f"{'band':>4}  {'luminance':>11}  {'percent':>13}  {'coverage':>8}  filename")
    for b in manifest.bands:
        lo_p, hi_p = b.pct_range
        lo_l, hi_l = b.luminance_range
        print(
            f"{b.index:>4}  {lo_l:>4} - {hi_l:>4}  "
            f"{lo_p:>5.1f} - {hi_p:>5.1f}%  {b.coverage_pct:>7.2f}%  {b.filename}"
        )
    if manifest.transparent_pct > 0:
        print(
            f"Ignored {manifest.transparent_pct:.2f}% of pixels "
            "(transparent in source)"
        )
    print(f"Manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
