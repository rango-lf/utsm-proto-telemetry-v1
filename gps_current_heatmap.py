"""GPS current heatmap renderer for UTSM Shell Eco-Marathon telemetry runs.

Usage
-----
python gps_current_heatmap.py Utsm.gpx telemetry_dumps/telemetry_20260411_112302.csv \\
    --laps 4 --split-method start --output outputs/current_heatmap.png

Produces one PNG per lap, colour-coded by current, acceleration, or
acceleration magnitude.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.collections import LineCollection
except ImportError as exc:
    raise SystemExit(
        "Missing required package. Install dependencies with: pip install matplotlib pandas numpy"
    ) from exc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utsm_telemetry import (
    FORWARD_AXIS_CHOICES,
    add_xy,
    align_telemetry,
    compute_lap_stats,
    find_lap_boundaries_by_start_gate,
    find_nearest_gps_index,
    find_start_spike,
    merge_by_time,
    parse_iso8601,
    parse_lap_time,
    read_gpx,
    read_telemetry,
    split_gps_into_laps,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align telemetry current data with a GPX track and render a heatmap."
    )
    parser.add_argument("gps", help="Path to the GPX track file")
    parser.add_argument("telemetry", help="Path to the telemetry CSV file")
    parser.add_argument(
        "--output", "-o",
        default="current_heatmap.png",
        help="Output image file for the heatmap",
    )
    parser.add_argument(
        "--metric",
        choices=["current", "accel", "magnitude"],
        default="current",
        help="Metric to colour the track by",
    )
    parser.add_argument(
        "--start-time",
        help=(
            "Force the telemetry start time to align with this GPS time. "
            "Use ISO 8601 (e.g. 2026-04-11T14:29:07Z). "
            "If omitted, telemetry row 0 is aligned to the first GPS point time."
        ),
    )
    parser.add_argument(
        "--time-offset-ms",
        type=float,
        default=0.0,
        help=(
            "Add a millisecond offset to the telemetry timestamps after aligning start. "
            "Useful when the telemetry stream begins slightly before or after the GPS track."
        ),
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=1.5,
        help="Maximum seconds to tolerate when matching telemetry rows to GPS track points.",
    )
    parser.add_argument(
        "--laps",
        type=int,
        default=1,
        help="Number of laps to split the GPS track into and render separately.",
    )
    parser.add_argument(
        "--split-method",
        choices=["points", "time", "line", "start"],
        default="start",
        help=(
            "How to split the GPS track into laps: by equal point count, equal time "
            "segments, by crossing a line, or by detecting the start/finish gate."
        ),
    )
    parser.add_argument(
        "--lap-times",
        nargs="+",
        metavar="ELAPSED",
        help=(
            "Elapsed time from the start of the Strava/GPX activity for each lap start. "
            "Use MM:SS or H:MM:SS (e.g. 11:20 or 1:02:30). Provide one per lap. "
            "The first time is also used to anchor telemetry: "
            "the first 10 A current spike is aligned to this point."
        ),
    )
    parser.add_argument(
        "--forward-axis",
        choices=FORWARD_AXIS_CHOICES,
        default="ax",
        help="IMU axis that points forward (used when --metric=accel).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Heatmap-specific helpers (not shared with other scripts)
# ---------------------------------------------------------------------------

def format_output_path(output: str, lap_index: int) -> str:
    base, ext = os.path.splitext(output)
    if not ext:
        ext = ".png"
    return f"{base}_lap{lap_index}{ext}"


def plot_heatmap(
    df: pd.DataFrame,
    color_column: str,
    output_path: str,
    lap_index: int | None = None,
    stats: dict | None = None,
) -> None:
    df = add_xy(df)
    coords = df[["x", "y"]].to_numpy()
    if len(coords) < 2:
        raise ValueError("Not enough merged points to draw a track")

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    values = df[color_column].to_numpy()
    norm = plt.Normalize(np.nanmin(values), np.nanmax(values))
    cmap = "plasma" if color_column == "current_mA" else "viridis"

    fig, ax = plt.subplots(figsize=(10, 8))
    line_collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=3,
        capstyle="round",
    )
    line_collection.set_array(values[:-1])
    ax.add_collection(line_collection)
    ax.autoscale()
    cbar = fig.colorbar(line_collection, ax=ax, pad=0.02)
    cbar.set_label(color_column)

    ax.scatter(df["x"].iloc[0], df["y"].iloc[0], color="green", marker="o", s=80, label="Start")
    ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="red", marker="X", s=80, label="End")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    title = f"Lap {lap_index} track heatmap" if lap_index is not None else "Track heatmap"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    if stats is not None:
        stat_text = (
            f"Duration: {stats['duration_s']:.1f}s\n"
            f"Distance: {stats['distance_m']:.1f}m\n"
            f"Avg current: {stats['avg_current_mA']:.1f}mA, max: {stats['max_current_mA']:.1f}mA\n"
            f"Avg accel: {stats['avg_accel_m_s2']:.2f}m/s², max: {stats['max_accel_m_s2']:.2f}m/s²"
        )
        ax.text(
            0.98,
            0.02,
            stat_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray", "boxstyle": "round,pad=0.3"},
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved heatmap to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    gps_df = read_gpx(args.gps)
    telem_df = read_telemetry(args.telemetry)

    color_column = {
        "current": "current_mA",
        "accel": "accel_m_s2",
        "magnitude": "accel_mag",
    }[args.metric]

    # --- Manual lap times mode ---
    if args.lap_times:
        track_start = gps_df["time"].iloc[0]
        lap_timestamps = [parse_lap_time(t, track_start) for t in args.lap_times]
        if len(lap_timestamps) < 2:
            raise ValueError("--lap-times requires at least 2 times: one lap start and a finish time.")
        print(f"GPX track starts at {track_start}. Lap boundaries resolved to:")
        for i, ts in enumerate(lap_timestamps):
            label = f"Lap {i + 1} start" if i < len(lap_timestamps) - 1 else "Finish"
            print(f"  {label}: {args.lap_times[i]} \u2192 {ts}")

        spike_idx = find_start_spike(telem_df)
        spike_ms = float(telem_df.loc[spike_idx, "timestamp_ms"])
        telemetry_start = lap_timestamps[0] - pd.Timedelta(milliseconds=spike_ms)
        print(
            f"Telemetry spike at row {spike_idx} (timestamp_ms={spike_ms:.0f}). "
            f"Aligning to lap 1 start: {lap_timestamps[0]}. "
            f"Computed telemetry epoch: {telemetry_start}."
        )
        telem_df = align_telemetry(telem_df, gps_df, telemetry_start, args.time_offset_ms)

        laps = []
        for i in range(len(lap_timestamps) - 1):
            lap_start = lap_timestamps[i]
            lap_end = lap_timestamps[i + 1]
            lap_gps = (
                gps_df[(gps_df["time"] >= lap_start) & (gps_df["time"] < lap_end)]
                .copy()
                .reset_index(drop=True)
            )
            laps.append(lap_gps)
            print(f"Lap {i + 1}: GPS points={len(lap_gps)} ({lap_start} \u2192 {lap_end})")

    elif args.split_method == "start":
        telem_df = align_telemetry(telem_df, gps_df, args.start_time, args.time_offset_ms)
        spike_idx = find_start_spike(telem_df)
        start_time = telem_df.loc[spike_idx, "time"]
        gps_start_idx = find_nearest_gps_index(gps_df, start_time)
        print(
            f"Start spike at telemetry index {spike_idx}, time {start_time}, "
            f"matching GPS index {gps_start_idx}."
        )
        gps_df = gps_df.loc[gps_start_idx:].reset_index(drop=True)
        # Use the accurate start-gate detector (same as analyze_strategy.py)
        lap_boundaries = find_lap_boundaries_by_start_gate(gps_df, 0, args.laps)
        if len(lap_boundaries) < args.laps + 1:
            print(
                f"Warning: only found {len(lap_boundaries) - 1} complete lap(s) via start-gate "
                f"detection (wanted {args.laps}). Remaining laps will use the final segment."
            )
        laps = []
        for i in range(min(len(lap_boundaries) - 1, args.laps)):
            laps.append(gps_df.iloc[lap_boundaries[i]: lap_boundaries[i + 1]].reset_index(drop=True))
        if len(laps) < args.laps and lap_boundaries:
            laps.append(gps_df.iloc[lap_boundaries[-1]:].reset_index(drop=True))
        if not laps:
            raise ValueError("Could not create any laps from the start point detection.")

    else:
        telem_df = align_telemetry(telem_df, gps_df, args.start_time, args.time_offset_ms)
        laps = split_gps_into_laps(gps_df, args.laps, args.split_method)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for idx, lap_gps in enumerate(laps, start=1):
        if lap_gps.empty:
            print(f"Skipping empty lap {idx}")
            continue

        lap_start = lap_gps["time"].iloc[0]
        lap_end = lap_gps["time"].iloc[-1]
        lap_telem = telem_df[
            (telem_df["time"] >= lap_start - pd.Timedelta(seconds=args.tolerance_sec))
            & (telem_df["time"] <= lap_end + pd.Timedelta(seconds=args.tolerance_sec))
        ].copy()

        if lap_telem.empty:
            print(f"Skipping lap {idx}: no telemetry rows fall within lap time range.")
            continue

        merged = merge_by_time(lap_telem, lap_gps, args.tolerance_sec)
        stats = compute_lap_stats(merged)
        print(
            f"Lap {idx}: duration={stats['duration_s']:.1f}s, distance={stats['distance_m']:.1f}m, "
            f"avg_current={stats['avg_current_mA']:.1f}mA, max_current={stats['max_current_mA']:.1f}mA, "
            f"avg_accel={stats['avg_accel_m_s2']:.2f}m/s², max_accel={stats['max_accel_m_s2']:.2f}m/s²"
        )

        output_path = format_output_path(args.output, idx) if len(laps) > 1 else args.output
        plot_heatmap(merged, color_column, output_path, lap_index=idx, stats=stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
