"""Render interactive forecast / backtest charts (Plotly HTML) for a run.

- ``kind="forecast"``: actuals (solid) + future P50/P90/P95 (dotted), with a shaded band.
- ``kind="backtest"``: actuals vs predicted P50/P90/P95 over a historical window.

Navigation: no Plotly range slider (it double-renders and drags slowly). Instead:
**quick-range buttons** under the legend set the window; a lightweight **jump bar** (HTML
slider below the chart) slides that window across the whole timeline via ``Plotly.relayout``
(no re-render); **drag to pan** and **mouse-wheel to zoom**. Double-click resets.

Performance: the (potentially huge) ``actual`` trace is **decimated** with LTTB
(shape-preserving) to ``max_points`` and drawn with **WebGL** (Scattergl). The small
quantile lines stay SVG to keep the dotted styling; the P50–P95 band is SVG fill.
Quantile colors: P50 blue, P90 orange, P95 green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asp_demand.config import backtest_path, forecast_path, parquet_path

DEFAULT_MAX_POINTS = 4000

_QUANTILE_STYLE = {
    "p50": {"color": "#3b82f6", "name": "P50"},  # blue
    "p90": {"color": "#f59e0b", "name": "P90"},  # orange
    "p95": {"color": "#22c55e", "name": "P95"},  # green
}

# Quick-range buttons (placed under the legend), per granularity.
_RANGE_BUTTONS: dict[str, list[dict[str, Any]]] = {
    "hourly": [
        {"count": 12, "label": "12h", "step": "hour", "stepmode": "backward"},
        {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
        {"count": 3, "label": "3d", "step": "day", "stepmode": "backward"},
        {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
        {"step": "all", "label": "all"},
    ],
    "daily": [
        {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
        {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
        {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
        {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
        {"step": "all", "label": "all"},
    ],
}


# A lightweight "jump bar": an HTML range input below the chart that slides the current
# x-axis window across the full data extent via Plotly.relayout (no trace re-render).
_JUMP_BAR_JS = """
(function() {
  var gd = document.getElementsByClassName('plotly-graph-div')[0];
  if (!gd) return;
  function extent() {
    var lo = Infinity, hi = -Infinity;
    (gd.data || []).forEach(function(t) {
      if (!t.x || !t.x.length) return;
      var a = new Date(t.x[0]).getTime(), b = new Date(t.x[t.x.length - 1]).getTime();
      if (a < lo) lo = a; if (b > hi) hi = b;
    });
    return [lo, hi];
  }
  var ext = extent();
  var wrap = document.createElement('div');
  wrap.style.cssText = 'padding:2px 12px 12px;font:12px sans-serif;color:#888';
  wrap.innerHTML = '<span>\\u23ee jump through time \\u23ed</span>';
  var bar = document.createElement('input');
  bar.type = 'range'; bar.min = '0'; bar.max = '1000'; bar.value = '1000';
  bar.style.cssText = 'width:100%';
  wrap.appendChild(bar);
  gd.parentNode.insertBefore(wrap, gd.nextSibling);
  bar.addEventListener('input', function() {
    var xa = gd._fullLayout && gd._fullLayout.xaxis;
    var rng = (xa && xa.range) ? xa.range : ext;
    var span = new Date(rng[1]).getTime() - new Date(rng[0]).getTime();
    var total = ext[1] - ext[0];
    if (!(span > 0) || span >= total) return;  // fully zoomed out: nothing to jump
    var start = ext[0] + (bar.value / 1000) * (total - span);
    Plotly.relayout(gd, {'xaxis.range': [
      new Date(start).toISOString(), new Date(start + span).toISOString()]});
  });
})();
"""


def _lttb_indices(x: pd.Series, y: pd.Series, max_points: int) -> np.ndarray:
    """Largest-Triangle-Three-Buckets downsample: indices of ~max_points to keep.

    Preserves visual shape (peaks/troughs) far better than uniform striding. First and
    last points are always kept. Returns all indices when no decimation is needed.
    """
    n = len(x)
    if max_points < 3 or n <= max_points:
        return np.arange(n)
    xi = x.astype("int64").to_numpy().astype(float)
    yi = y.to_numpy(dtype=float)
    bucket = (n - 2) / (max_points - 2)
    keep = np.empty(max_points, dtype=int)
    keep[0] = 0
    a = 0
    for i in range(max_points - 2):
        start = int(i * bucket) + 1
        end = int((i + 1) * bucket) + 1
        # next bucket, used for the averaged "third" triangle vertex
        nstart = min(end, n - 1)
        nend = max(min(int((i + 2) * bucket) + 1, n), nstart + 1)
        avg_x, avg_y = xi[nstart:nend].mean(), yi[nstart:nend].mean()
        seg_x, seg_y = xi[start:end], yi[start:end]
        if seg_x.size == 0:
            chosen = start
        else:
            areas = np.abs((xi[a] - avg_x) * (seg_y - yi[a]) - (xi[a] - seg_x) * (avg_y - yi[a]))
            chosen = start + int(np.argmax(areas))
        keep[i + 1] = chosen
        a = chosen
    keep[-1] = n - 1
    return keep


def _decimate(df: pd.DataFrame, value_col: str, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = _lttb_indices(df["bucket_start"], df[value_col], max_points)
    return df.iloc[idx]


def _add_quantiles(fig: Any, df: pd.DataFrame) -> None:
    """Add the P50/P90/P95 dotted SVG lines (+ faint P50–P95 band) from a frame."""
    import plotly.graph_objects as go

    if "p95" in df and "p50" in df:
        fig.add_trace(go.Scatter(x=df["bucket_start"], y=df["p95"], mode="lines",
                                 line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["bucket_start"], y=df["p50"], mode="lines",
                                 line={"width": 0}, fill="tonexty",
                                 fillcolor="rgba(59,130,246,0.10)", showlegend=False,
                                 hoverinfo="skip"))
    for label, style in _QUANTILE_STYLE.items():
        if label in df:
            fig.add_trace(go.Scatter(
                x=df["bucket_start"], y=df[label], mode="lines", name=style["name"],
                line={"color": style["color"], "dash": "dot"},
            ))


def plot_forecast(
    run_dir: str | Path, granularity: str, out: str | Path | None = None,
    height: int = 900, kind: str = "forecast", max_points: int = DEFAULT_MAX_POINTS,
) -> Path:
    """Write an interactive HTML chart for ``granularity`` in ``run_dir``; return its path.

    ``kind="forecast"`` overlays actuals with the future quantiles; ``kind="backtest"``
    overlays actuals with predicted quantiles over a historical window. The source CSV
    must exist (run ``predict`` / ``backtest`` first). ``max_points`` decimates the actual
    trace for rendering speed.
    """
    import plotly.graph_objects as go

    if kind not in ("forecast", "backtest"):
        raise ValueError(f"unknown kind {kind!r}; choose forecast or backtest")
    rd = Path(run_dir)
    src = (forecast_path if kind == "forecast" else backtest_path)(rd, granularity)
    if not src.exists():
        verb = "predict" if kind == "forecast" else "backtest"
        raise FileNotFoundError(f"no {kind} at {src}; run `asp-demand {verb}` first")
    data = pd.read_csv(src, parse_dates=["bucket_start"])

    fig: Any = go.Figure()
    if kind == "forecast":
        hist_path = parquet_path(rd, granularity)
        if hist_path.exists():
            actual = _decimate(
                pd.read_parquet(hist_path).sort_values("bucket_start"), "request_count", max_points
            )
            # the heavy trace: WebGL so it stays smooth even at the cap
            fig.add_trace(go.Scattergl(x=actual["bucket_start"], y=actual["request_count"],
                                       mode="lines", name="actual"))
            if not actual.empty:
                fig.add_vline(x=actual["bucket_start"].max(), line_dash="dot", line_color="gray")
        _add_quantiles(fig, _decimate(data, "p50", max_points))
        title = f"{granularity} forecast — {rd.name}"
    else:  # backtest: actual + predicted over the same (decimated) window
        data = _decimate(data, "actual", max_points)
        fig.add_trace(go.Scattergl(x=data["bucket_start"], y=data["actual"],
                                   mode="lines", name="actual"))
        _add_quantiles(fig, data)
        title = f"{granularity} backtest (predicted vs actual) — {rd.name}"

    fig.update_layout(
        title=title,
        yaxis_title="request_count",
        hovermode="x unified",
        height=height,
        dragmode="pan",  # drag scrolls through dates (keeps the zoom window); wheel zooms
        # legend on top, quick-range buttons just beneath it
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.16, "xanchor": "left", "x": 0},
        margin={"t": 120},
    )
    fig.update_xaxes(
        title="time",
        rangeselector={
            "buttons": _RANGE_BUTTONS.get(granularity, _RANGE_BUTTONS["daily"]),
            "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom",
        },
    )

    out_path = (rd / f"{kind}_{granularity}.html") if out is None else Path(out)
    fig.write_html(
        str(out_path), default_height=f"{height}px",
        config={"scrollZoom": True, "displaylogo": False},
        post_script=_JUMP_BAR_JS,
    )
    return out_path
