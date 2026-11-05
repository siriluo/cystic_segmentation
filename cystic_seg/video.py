"""Video utilities: clip by time, extract frames, stitch frames (ffmpeg, no OpenCV)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


FFMPEG_FALLBACKS = (
    "/sw/spack/v1/apps/ffmpeg/6.1.1-gcc-13.2.1.native-sq7jsv4/bin/ffmpeg",
)


def parse_time(timestr: str) -> int:
    """Parse hh:mm:ss, mm:ss, or ss into seconds."""
    parts = [int(p) for p in timestr.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    elif len(parts) == 1:
        h = 0
        m = 0
        s = parts[0]
    else:
        raise ValueError("Invalid time format. Use hh:mm:ss, mm:ss, or ss.")
    return h * 3600 + m * 60 + s


def _which_tool(name: str) -> str | None:
    env_key = name.upper()
    env = os.environ.get(env_key)
    if env and Path(env).is_file():
        return env
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        for candidate in FFMPEG_FALLBACKS:
            if Path(candidate).is_file():
                return candidate
        ffprobe_env = os.environ.get("FFPROBE")
        if ffprobe_env:
            sibling = Path(ffprobe_env).with_name("ffmpeg")
            if sibling.is_file():
                return str(sibling)
    if name == "ffprobe":
        ffmpeg = _which_tool("ffmpeg")
        if ffmpeg:
            sibling = Path(ffmpeg).with_name("ffprobe")
            if sibling.is_file():
                return str(sibling)
    return None


def require_ffmpeg() -> str:
    path = _which_tool("ffmpeg")
    if not path:
        raise SystemExit(
            "ffmpeg not found. On DeltaAI run `module load ffmpeg/6.1.1`, "
            "or set FFMPEG to the ffmpeg binary."
        )
    return path


def require_ffprobe() -> str:
    path = _which_tool("ffprobe")
    if not path:
        ffmpeg = require_ffmpeg()
        sibling = Path(ffmpeg).with_name("ffprobe")
        if sibling.is_file():
            return str(sibling)
        raise SystemExit("ffprobe not found. module load ffmpeg/6.1.1 or set FFPROBE.")
    return path


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-4000:]}")


def probe_video(path: str | Path) -> dict[str, Any]:
    """Return width, height, fps (float), fps_str, nb_frames, duration."""
    cmd = [
        require_ffprobe(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr}")
    info = json.loads(proc.stdout)
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise RuntimeError(f"No video stream in {path}")
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    if rate in ("0/0", "0", "", None):
        rate = stream.get("r_frame_rate") or "30/1"
    try:
        fps = float(Fraction(rate))
    except (ZeroDivisionError, ValueError):
        fps = 30.0
        rate = "30/1"
    nb = stream.get("nb_frames")
    try:
        n_frames = int(nb) if nb not in (None, "N/A") else None
    except (TypeError, ValueError):
        n_frames = None
    duration = info.get("format", {}).get("duration")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "fps_str": str(rate),
        "nb_frames": n_frames,
        "duration": float(duration) if duration is not None else None,
    }


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    max_frames: int | None = None,
    quality: int = 3,
) -> list[Path]:
    """Write sequential JPEGs `000001.jpg`, ... and return sorted paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(out / "%06d.jpg")
    cmd = [require_ffmpeg(), "-y", "-i", str(video)]
    if max_frames is not None:
        cmd += ["-vframes", str(int(max_frames))]
    cmd += ["-q:v", str(quality), "-start_number", "1", pattern]
    _run(cmd)
    frames = sorted(out.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg wrote no frames to {out}")
    return frames


def stitch_frames(
    frame_dir: str | Path,
    output: str | Path,
    fps_str: str = "30/1",
    pattern: str = "%06d.jpg",
) -> None:
    """Encode sequential JPEGs into an MP4 (mpeg4; this ffmpeg build has no libx264)."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    src = Path(frame_dir) / pattern
    ffmpeg = require_ffmpeg()
    attempts = [
        [ffmpeg, "-y", "-framerate", str(fps_str), "-i", str(src), "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", str(out)],
        [ffmpeg, "-y", "-framerate", str(fps_str), "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)],
        [ffmpeg, "-y", "-framerate", str(fps_str), "-i", str(src), "-c:v", "mpeg4", "-q:v", "5", str(out)],
    ]
    errors: list[str] = []
    for cmd in attempts:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            return
        errors.append(proc.stderr[-800:])
    raise RuntimeError("ffmpeg could not encode the overlay video:\n" + "\n---\n".join(errors))


def chunk_video(input_file: str | Path, output_file: str | Path, start_time: float, end_time: float) -> None:
    """Clip `[start_time, end_time)` seconds with ffmpeg (falls back to OpenCV)."""
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _which_tool("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-i",
            str(input_file),
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-an",
            str(out),
        ]
        _run(cmd)
        print(f"Saved chunked video to {output_file} from {start_time}s to {end_time}s.")
        return

    import cv2

    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open input video: {input_file}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    try:
        while frame_idx < end_frame and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    print(f"Saved chunked video to {output_file} from {start_time}s to {end_time}s.")
