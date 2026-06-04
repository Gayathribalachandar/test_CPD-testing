import argparse
from pathlib import Path

from animate_cpd import (
    build_animation,
    default_video_output_path,
    load_pos_history,
    save_animation_video,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render CPD position history to an MP4 video.")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory containing output/pos_history.npy. Defaults to CPD_WORKSPACE_DIR or project workspace.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Defaults to <workspace>/output/cpd_animation.mp4.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video frames per second.")
    parser.add_argument("--dpi", type=int, default=200, help="Video render DPI.")
    parser.add_argument("--interval-ms", type=int, default=50, help="Animation frame interval in milliseconds.")
    parser.add_argument("--marker-size", type=float, default=0.5, help="Scatter marker size.")
    return parser.parse_args()


def main():
    args = parse_args()
    pos_history, source_path = load_pos_history(workspace_dir=args.workspace)
    workspace_dir = Path(args.workspace) if args.workspace else None
    output_path = Path(args.output) if args.output else default_video_output_path(source_path, workspace_dir)

    fig, _ax, ani = build_animation(
        pos_history,
        interval=args.interval_ms,
        marker_size=args.marker_size,
    )
    save_animation_video(ani, output_path, fps=args.fps, dpi=args.dpi)
    fig.clf()
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
