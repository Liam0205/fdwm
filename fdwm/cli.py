import argparse
from pathlib import Path
import sys
from typing import List

import fdwm
from fdwm.watermark import embed, extract, extract_text

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def _gather_images(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    imgs: List[Path] = []
    for p in target.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            imgs.append(p)
    return imgs


def _cmd_embed(args):
    images = _gather_images(args.host)
    if not images:
        print("No images found to process.", file=sys.stderr)
        sys.exit(1)
    print(f"Embedding watermark into {len(images)} image(s)...")
    for img_path in images:
        try:
            out_path, metrics = embed(
                host_path=img_path,
                watermark_path=str(args.wm_img) if args.wm_img else None,
                output_path=img_path,  # overwrite
                watermark_text=args.wm_text,
                strength=args.strength,
                scale=args.scale,
                font_path=str(args.font) if args.font else None,
                font_size=args.font_size,
                debug=args.debug,
                grid_m=args.grid_m,
                grid_n=args.grid_n,
            )
            print(f"✔ Watermark applied: {img_path}")
            if args.debug:
                print(f"  Mean pixel diff: {metrics['mean_pixel_diff']:.2f}")
                print(f"  Max pixel diff: {metrics['max_pixel_diff']:.2f}")
                print(f"  90th percentile pixel diff: {metrics['p90_pixel_diff']:.2f}")
                print(f"  PSNR: {metrics['psnr']:.2f} dB")
        except Exception as e:
            print(f"✖ Failed to watermark {img_path}: {e}", file=sys.stderr)


def _cmd_extract(args):
    images = _gather_images(args.input)
    if not images:
        print("No images found to process.", file=sys.stderr)
        sys.exit(1)
    out_dir = args.output or args.input if args.input.is_dir() else args.input.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting watermark from {len(images)} image(s)...")
    for img_path in images:
        try:
            stem = img_path.stem
            if args.text:
                txt = extract_text(
                    watermarked_path=img_path,
                    strength=args.strength,
                    scale=args.scale,
                    grid_m=args.grid_m,
                    grid_n=args.grid_n,
                )
                print(f"[{img_path}] -> {txt}")
                if args.save_text:
                    (out_dir / f"{stem}_wm.txt").write_text(txt, encoding="utf-8")
            else:
                out_path = out_dir / f"{stem}_wm.png"
                extract(
                    watermarked_path=img_path,
                    strength=args.strength,
                    scale=args.scale,
                    output_path=out_path,
                    grid_m=args.grid_m,
                    grid_n=args.grid_n,
                )
                print(f"✔ Watermark image saved: {out_path}")
        except Exception as e:
            print(f"✖ Failed to extract from {img_path}: {e}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdwm", description="Frequency-domain watermark CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # embed subcommand
    p_embed = sub.add_parser("embed", help="Embed watermark into image(s)")
    p_embed.add_argument(
        "host", type=Path, help="Image file or directory to embed watermark"
    )

    wm_group = p_embed.add_mutually_exclusive_group(required=True)
    wm_group.add_argument(
        "--watermark-img", type=Path, dest="wm_img", help="Watermark image path"
    )
    wm_group.add_argument(
        "--watermark-text", type=str, dest="wm_text", help="Watermark text"
    )

    p_embed.add_argument("--strength", type=float, default=30000.0)
    p_embed.add_argument("--scale", type=float, default=0.25)
    p_embed.add_argument("--font", type=Path)
    p_embed.add_argument("--font-size", type=int)
    p_embed.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed metrics for each processed image",
    )
    p_embed.add_argument(
        "--grid-m",
        type=int,
        default=3,
        help="Number of vertical grid divisions (default: 3)",
    )
    p_embed.add_argument(
        "--grid-n",
        type=int,
        default=3,
        help="Number of horizontal grid divisions (default: 3)",
    )
    p_embed.set_defaults(func=_cmd_embed)

    # extract subcommand
    p_ext = sub.add_parser("extract", help="Extract watermark from image(s)")
    p_ext.add_argument("input", type=Path, help="Watermarked image file or directory")
    p_ext.add_argument("--strength", type=float, default=30000.0)
    p_ext.add_argument("--scale", type=float, default=0.25)
    p_ext.add_argument(
        "--output", type=Path, help="Directory to save extracted watermark images/text"
    )
    p_ext.add_argument(
        "--text",
        action="store_true",
        help="Perform OCR and output text instead of image",
    )
    p_ext.add_argument(
        "--save-text", action="store_true", help="Save recognized text to .txt files"
    )
    p_ext.add_argument(
        "--grid-m",
        type=int,
        default=3,
        help="Number of vertical grid divisions used during embedding (default: 3)",
    )
    p_ext.add_argument(
        "--grid-n",
        type=int,
        default=3,
        help="Number of horizontal grid divisions used during embedding (default: 3)",
    )
    p_ext.set_defaults(func=_cmd_extract)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
