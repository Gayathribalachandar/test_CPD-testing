import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers


def resolve_pos_history_file(workspace_dir=None):
    project_root = Path(__file__).resolve().parents[2]
    if workspace_dir is None:
        workspace_dir = Path(os.environ.get("CPD_WORKSPACE_DIR", str(project_root / "workspace")))
    else:
        workspace_dir = Path(workspace_dir)
    workspace_output_path = workspace_dir / "output" / "pos_history.npy"
    workspace_legacy_path = workspace_dir / "pos_history.npy"
    legacy_path = Path(__file__).resolve().parent / "pos_history.npy"
    for candidate in (workspace_output_path, workspace_legacy_path, legacy_path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "pos_history.npy not found. Checked: "
        f"{workspace_output_path}, {workspace_legacy_path}, and {legacy_path}"
    )


def load_pos_history(workspace_dir=None):
    pos_history_file = resolve_pos_history_file(workspace_dir=workspace_dir)
    return np.load(str(pos_history_file)), pos_history_file


def build_animation(pos_history, interval=50, marker_size=0.5):
    n_snaps, _n_nodes, _ = pos_history.shape
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=marker_size)
    ax.set_aspect("equal", "box")

    x_min, x_max = pos_history[:, :, 0].min(), pos_history[:, :, 0].max()
    y_min, y_max = pos_history[:, :, 1].min(), pos_history[:, :, 1].max()
    margin = 0.05
    x_span = max(float(x_max - x_min), 1e-9)
    y_span = max(float(y_max - y_min), 1e-9)
    ax.set_xlim(x_min - margin * x_span, x_max + margin * x_span)
    ax.set_ylim(y_min - margin * y_span, y_max + margin * y_span)

    title = ax.set_title("")

    def update(frame):
        scat.set_offsets(pos_history[frame])
        title.set_text(f"Step {frame + 1}/{n_snaps}")
        return scat, title

    ani = FuncAnimation(fig, update, frames=n_snaps, interval=interval, blit=False)
    return fig, ax, ani


def default_video_output_path(source_path, workspace_dir=None):
    if workspace_dir is not None:
        return Path(workspace_dir) / "output" / "cpd_animation.mp4"
    if source_path.name == "pos_history.npy" and source_path.parent.name == "output":
        return source_path.parent / "cpd_animation.mp4"
    if source_path.parent.name == "workspace":
        return source_path.parent / "output" / "cpd_animation.mp4"
    return source_path.parent / "cpd_animation.mp4"


def save_animation_video(ani, output_path, fps=20, dpi=200):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is not available for Matplotlib video export.")
    writer = FFMpegWriter(fps=max(1, int(fps)))
    ani.save(str(output_path), writer=writer, dpi=max(72, int(dpi)))
    return output_path


def main():
    pos_history, source_path = load_pos_history()
    fig, _ax, ani = build_animation(pos_history)
    output_path = default_video_output_path(source_path)
    try:
        saved = save_animation_video(ani, output_path)
        print(f"Saved video to {saved}")
    except Exception as exc:
        print(f"Video export skipped: {exc}")
    plt.show()


if __name__ == "__main__":
    main()
