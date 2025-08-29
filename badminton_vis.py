"""
badminton_vis.py

A small library to load, process, and visualize 3D badminton shuttle trajectories in Jupyter using Plotly.
Meets the functional requirements described in the "羽毛球3D轨迹可视化" spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as e:
    raise RuntimeError(
        "Plotly is required for interactive 3D visualization. Please install plotly."
    ) from e


# -----------------------------
# Court Dimensions (BWF Standard)
# -----------------------------
@dataclass
class CourtSpec:
    # BWF standard metrics in meters (defaults)
    length: float = 13.40 * 100  # total court length
    width_doubles: float = 6.10 * 100  # total width for doubles
    width_singles: float = (
        5.18 * 100
    )  # total width for singles (reference; we draw full doubles by default)
    net_height_center: float = 1.524 * 100  # net height at center
    line_width: float = 0.04 * 100  # typical painted line width (visual only)
    # Service lines (distances from the net or back boundary)
    short_service_from_net: float = 1.98 * 100
    long_service_from_back_doubles: float = 0.76 * 100

    min_x_ground: float = length
    min_y_ground: float = width_doubles + 1 * 100
    min_z_ground: float = 8 * 100

    def half_length(self) -> float:
        return self.length / 2.0

    def half_width(self) -> float:
        return self.width_doubles / 2.0


# -----------------------------
# Data Model
# -----------------------------
@dataclass
class TrajectoryPoint:
    frame_id: Optional[int] = None
    ts: Optional[float] = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    pid: Union[str, int]
    msgtype: str
    points: List[TrajectoryPoint]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self.points:
            rows.append(
                {
                    "pid": self.pid,
                    "frame_id": p.frame_id,
                    "ts": p.ts,
                    "x": p.x,
                    "y": p.y,
                    "z": p.z,
                    "vx": p.vx,
                    "vy": p.vy,
                    "vz": p.vz,
                }
            )
        df = pd.DataFrame(rows)
        # sort by ts if present, else frame_id
        if "ts" in df.columns and df["ts"].notna().any():
            df = df.sort_values(["pid", "ts", "frame_id"], na_position="last")
        else:
            df = df.sort_values(["pid", "frame_id"])
        df = df.reset_index(drop=True)
        return df


# -----------------------------
# Parsing / Loading
# -----------------------------
def _parse_positions(positions: Iterable[Dict[str, Any]]) -> List[TrajectoryPoint]:
    pts: List[TrajectoryPoint] = []
    for item in positions:
        pos = item.get("pos") or {}
        p = TrajectoryPoint(
            frame_id=item.get("frame_id"),
            ts=float(item["ts"]) if "ts" in item and item["ts"] is not None else None,
            x=float(pos.get("x", np.nan)),
            y=float(pos.get("y", np.nan)),
            z=float(pos.get("z", np.nan)),
            vx=(float(item["vx"]) if "vx" in item and item["vx"] is not None else None),
            vy=(float(item["vy"]) if "vy" in item and item["vy"] is not None else None),
            vz=(float(item["vz"]) if "vz" in item and item["vz"] is not None else None),
            raw=item,
        )
        pts.append(p)
    return pts


def load_from_json_obj(
    obj: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> List[Trajectory]:
    """
    Accepts a single message or a list of messages following the schema:
    {
      "msgtype": "pred_track",
      "pid": "abc",
      "positions": [ { "frame_id": ..., "pos": {"x":..,"y":..,"z":..}, "ts": ... }, ... ],
      "timestamp": ...
    }
    Returns a list of Trajectory.
    """
    messages: List[Dict[str, Any]] = obj if isinstance(obj, list) else [obj]
    trajs: List[Trajectory] = []
    for msg in messages:
        pid = msg.get("pid", "unknown")
        msgtype = msg.get("msgtype", "unknown")
        positions = msg.get("positions", [])
        pts = _parse_positions(positions)
        trajs.append(Trajectory(pid=pid, msgtype=msgtype, points=pts))
    return trajs


def load_from_json_file(path: str) -> List[Trajectory]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return load_from_json_obj(obj)


# -----------------------------
# Preprocessing helpers
# -----------------------------
def sort_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure numeric
    for c in ["x", "y", "z", "ts", "vx", "vy", "vz"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Sort
    if "ts" in df.columns and df["ts"].notna().any():
        df = df.sort_values(["pid", "ts", "frame_id"], na_position="last")
    else:
        df = df.sort_values(["pid", "frame_id"])
    df = df.reset_index(drop=True)
    # Drop rows with missing essential coords
    df = df.dropna(subset=["x", "y", "z"])
    return df


def smooth(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Simple moving average smoothing over x,y,z per pid."""
    df = df.copy()
    for pid, idx in df.groupby("pid").groups.items():
        for col in ["x", "y", "z"]:
            df.loc[idx, col] = (
                df.loc[idx, col].rolling(window, min_periods=1, center=True).mean()
            )
    return df


def interpolate_time(df: pd.DataFrame, freq_ms: int = 10) -> pd.DataFrame:
    """Linear interpolation to a uniform time grid per pid (requires 'ts' in milliseconds)."""
    if "ts" not in df.columns or df["ts"].isna().all():
        raise ValueError(
            "interpolate_time requires a 'ts' column with timestamps (ms)."
        )
    out = []
    for pid, g in df.groupby("pid"):
        g = g.sort_values("ts")
        # create target time grid
        ts_min, ts_max = g["ts"].min(), g["ts"].max()
        ts_target = np.arange(ts_min, ts_max + freq_ms, freq_ms)
        gi = pd.DataFrame({"ts": ts_target})
        # merge and interpolate
        merged = pd.merge_asof(
            gi,
            g[["ts", "x", "y", "z", "vx", "vy", "vz"]].sort_values("ts"),
            on="ts",
            direction="nearest",
        )
        merged["pid"] = pid
        out.append(merged)
    out = pd.concat(out, ignore_index=True)
    # optional interpolation for NaNs
    out[["x", "y", "z"]] = out[["x", "y", "z"]].interpolate(
        method="linear", limit_direction="both"
    )
    return out


# -----------------------------
# Court drawing (Plotly traces)
# -----------------------------
def court_traces(
    spec: CourtSpec = CourtSpec(),
    show_service_lines: bool = True,
    show_center_line: bool = True,
    surface_opacity: float = 0.08,
    surface_color: str = "green",
    line_color: str = "white",
    line_width: int = 3,
) -> List[go.Scatter3d]:
    """
    Returns 3D scatter traces that draw a standard badminton court on z=0 and a vertical net at x=0.
    """
    traces: List[go.Scatter3d] = []
    L = spec.length
    W = spec.width_doubles
    hl = spec.half_length()
    hw = spec.half_width()

    # Court rectangle (filled surface at z=0) using a dense outline

    # Shift all x by hl to move origin to baseline center
    court_outline = np.array(
        [[L, -hw, 0], [0, -hw, 0], [0, hw, 0], [L, hw, 0], [L, -hw, 0]]
    )
    traces.append(
        go.Scatter3d(
            x=court_outline[:, 0],
            y=court_outline[:, 1],
            z=court_outline[:, 2],
            mode="lines",
            line=dict(width=line_width, color=line_color),
            name="Court Boundary",
        )
    )

    # Surface mesh (lightly colored rectangle at z=0)
    surface = go.Mesh3d(
        x=[L, 0, 0, L],
        y=[-hw, -hw, hw, hw],
        z=[0, 0, 0, 0],
        i=[0, 1, 2],
        j=[0, 2, 3],
        k=[0, 3, 1],
        opacity=surface_opacity,
        color=surface_color,
        name="Court Surface",
        showscale=False,
    )
    traces.append(surface)

    # Back boundary (redundant because of outline, but keeps visuals crisp)
    # Service short lines (distance from net)
    if show_service_lines:
        d = spec.short_service_from_net
        # Short service lines
        traces.append(
            go.Scatter3d(
                x=[L - (hl + d), L - (hl + d)],
                y=[-hw, hw],
                z=[0, 0],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name="Short Service Line",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[L - (hl - d), L - (hl - d)],
                y=[-hw, hw],
                z=[0, 0],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name="Short Service Line",
            )
        )

        # Long service line for doubles
        traces.append(
            go.Scatter3d(
                x=[
                    L - (L - spec.long_service_from_back_doubles),
                    L - (L - spec.long_service_from_back_doubles),
                ],
                y=[-hw, hw],
                z=[0, 0],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name="Long Service Line (Doubles)",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[
                    L - spec.long_service_from_back_doubles,
                    L - spec.long_service_from_back_doubles,
                ],
                y=[-hw, hw],
                z=[0, 0],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name="Long Service Line (Doubles)",
            )
        )

    # Center service line (y=0) between short service line and back, per side
    if show_center_line:
        d = spec.short_service_from_net
        # Center line from short service line to back boundary, both sides
        traces.append(
            go.Scatter3d(
                x=[L - (hl + d), 0],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name="Center Line",
            )
        )
        traces.append(
            go.Scatter3d(
                x=[L - (hl - d), L],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name="Center Line",
            )
        )

    # Net (vertical rectangle at x=0)
    # Net at x=L-hl
    net = go.Mesh3d(
        x=[L - hl, L - hl, L - hl, L - hl],
        y=[-hw, hw, hw, -hw],
        z=[0, 0, spec.net_height_center, spec.net_height_center],
        i=[0, 1, 2],
        j=[0, 2, 3],
        k=[0, 3, 1],
        opacity=0.4,
        color="gray",
        name="Net",
        showscale=False,
    )
    traces.append(net)
    return traces


# -----------------------------
# Trajectory visualization
# -----------------------------
def trajectories_traces(
    df: pd.DataFrame,
    pid_palette: Optional[Dict[Union[str, int], str]] = None,
    line_width: int = 4,
    show_markers: bool = False,
    marker_size: int = 3,
) -> List[go.Scatter3d]:
    """
    Build static trajectory line traces for each pid.
    """
    traces: List[go.Scatter3d] = []
    for pid, g in df.groupby("pid"):
        g = g.sort_values(["ts", "frame_id"], na_position="last")
        color = None
        if pid_palette and pid in pid_palette:
            color = pid_palette[pid]
        traces.append(
            go.Scatter3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                mode="lines+markers" if show_markers else "lines",
                line=dict(width=line_width, color=color),
                marker=dict(size=marker_size) if show_markers else None,
                name=f"PID {pid}",
            )
        )
    return traces


def make_figure(
    df: pd.DataFrame,
    court: CourtSpec = CourtSpec(),
    title: str = "Badminton 3D Trajectories",
    pid_palette: Optional[Dict[Union[str, int], str]] = None,
    show_markers: bool = False,
) -> go.Figure:
    traces = []
    traces += court_traces(court)
    traces += trajectories_traces(
        df, pid_palette=pid_palette, show_markers=show_markers
    )

    fig = go.Figure(data=traces)

    # 限制最小可见场地范围（单位：cm）
    min_x, max_x = 0, court.min_x_ground
    min_y, max_y = court.min_y_ground / -2, court.min_y_ground / 2
    min_z, max_z = 0, court.min_z_ground
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (cm)",
            yaxis_title="Y (cm)",
            zaxis_title="Z (cm)",
            aspectmode="data",
            xaxis=dict(showbackground=True, zeroline=True, range=[min_x, max_x]),
            yaxis=dict(showbackground=True, zeroline=True, range=[min_y, max_y]),
            zaxis=dict(showbackground=True, zeroline=True, range=[min_z, max_z]),
        ),
        legend=dict(orientation="h"),
    )
    return fig


# -----------------------------
# Animation
# -----------------------------
def make_animation(
    df: pd.DataFrame,
    court: CourtSpec = CourtSpec(),
    title: str = "Badminton 3D Trajectory Animation",
    time_col: str = "ts",
    frame_step: int = 1,
    show_trails: bool = True,
) -> go.Figure:
    """
    Create a Plotly animation. If 'ts' exists, use it; otherwise fall back to index/frame_id.
    """
    df = df.copy()
    if time_col not in df.columns or df[time_col].isna().all():
        # create a synthetic timeline using row index per pid
        df["__t"] = df.groupby("pid").cumcount()
        time_col = "__t"

    # Normalize time to sorted unique keys
    times = np.sort(df[time_col].dropna().unique())
    times = times[:: max(1, frame_step)]

    # Base figure: court + (optional) full thin lines for reference
    court_parts = court_traces(court)
    base_traces = trajectories_traces(df, line_width=2, show_markers=False)
    fig = go.Figure(data=court_parts + base_traces)

    frames = []
    for t in times:
        frame_data = []
        # one marker per pid at current time, plus (optional) trail up to t
        for pid, g in df.groupby("pid"):
            g = g.sort_values(time_col)
            g_up_to_t = g[g[time_col] <= t]
            if g_up_to_t.empty:
                continue
            # Trail
            if show_trails:
                frame_data.append(
                    go.Scatter3d(
                        x=g_up_to_t["x"],
                        y=g_up_to_t["y"],
                        z=g_up_to_t["z"],
                        mode="lines",
                        line=dict(width=5),
                        name=f"Trail {pid}",
                        showlegend=False,
                    )
                )
            # Head marker
            last = g_up_to_t.iloc[-1]
            hovertext = (
                f"pid={pid}<br>t={last.get(time_col):.1f}"
                if pd.notna(last.get(time_col))
                else f"pid={pid}"
            )
            if "vx" in g.columns and "vy" in g.columns and "vz" in g.columns:
                if (
                    pd.notna(last.get("vx"))
                    and pd.notna(last.get("vy"))
                    and pd.notna(last.get("vz"))
                ):
                    speed = float(
                        np.sqrt(last["vx"] ** 2 + last["vy"] ** 2 + last["vz"] ** 2)
                    )
                    hovertext += f"<br>speed={speed:.2f} m/s"
            frame_data.append(
                go.Scatter3d(
                    x=[last["x"]],
                    y=[last["y"]],
                    z=[last["z"]],
                    mode="markers",
                    marker=dict(size=6),
                    name=f"Ball {pid}",
                    hovertext=hovertext,
                    hoverinfo="text",
                    showlegend=False,
                )
            )
        frames.append(go.Frame(data=frame_data, name=str(t)))

    # 限制最小可见场地范围（单位：cm）
    min_x, max_x = 0, court.min_x_ground
    min_y, max_y = court.min_y_ground / -2, court.min_y_ground / 2
    min_z, max_z = 0, court.min_z_ground
    fig.frames = frames
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            xaxis=dict(range=[min_x, max_x]),
            yaxis=dict(range=[min_y, max_y]),
            zaxis=dict(range=[min_z, max_z]),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 30, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": f"{t:.0f}",
                        "method": "animate",
                    }
                    for t in times
                ]
            }
        ],
    )
    return fig


# -----------------------------
# Export helpers
# -----------------------------
def save_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")


# -----------------------------
# Example utilities
# -----------------------------
def make_parabola_example(
    pid: str = "demo1",
    apex: Tuple[float, float, float] = (0.0, 0.0, 6.0),
    span_x: float = 6.0,
    steps: int = 120,
    duration_ms: int = 2400,
) -> List[Trajectory]:
    """
    Create a synthetic parabolic arc across the net to test visuals.
    """
    hl = span_x / 2.0
    xs = np.linspace(-hl, hl, steps)
    # Simple parabola with apex at (0,0,apex_z)
    ax, ay, az = apex
    # Parabola: z = az - a*x^2 - b*y^2, here y=0
    a = az / (hl**2 + 1e-9)
    zs = az - a * (xs - ax) ** 2
    ys = np.zeros_like(xs) + ay

    ts0 = 0.0
    dt = duration_ms / max(1, (steps - 1))
    ts = np.array([ts0 + i * dt for i in range(steps)])

    pts = []
    for i in range(steps):
        pts.append(
            TrajectoryPoint(
                frame_id=i,
                ts=float(ts[i]),
                x=float(xs[i]),
                y=float(ys[i]),
                z=float(max(0.05, zs[i])),
            )
        )
    return [Trajectory(pid=pid, points=pts)]
