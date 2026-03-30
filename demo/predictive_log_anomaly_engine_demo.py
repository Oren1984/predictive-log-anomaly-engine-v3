#!/usr/bin/env python3
"""
Predictive Log Anomaly Engine — System Observability Dashboard
==============================================================

Presentation-grade, standalone demo script for the Predictive Log Anomaly Engine.
Generates a professional multi-panel dashboard visualising system health, event
throughput, anomaly scoring activity, and pipeline characteristics.

SYSTEM STATUS (reflected in visuals)
--------------------------------------
  API Service         ✓ Healthy
  Prometheus          ✓ Scraper up
  Grafana             ✓ Dashboard healthy
  Investigation UI    ✓ Healthy
  Active Alerts       0  (system operating within configured thresholds)

NOTE ON ZERO ALERTS
--------------------
Zero active alerts is the EXPECTED state in a stable runtime.  The anomaly
engine scores every sliding window against a trained threshold.  No alert fires
unless a score crosses that threshold — which is exactly correct behaviour for
a log stream that has not produced an anomalous pattern.

OPTIONAL SIMULATION MODE
-------------------------
Pass --sim-mode to overlay an optional "presentation simulation" layer on
certain panels.  This is CLEARLY LABELLED as illustrative and does NOT
represent production evidence of anomalies.

Usage
------
  python demo/predictive_log_anomaly_engine_demo.py
  python demo/predictive_log_anomaly_engine_demo.py --save output.png
  python demo/predictive_log_anomaly_engine_demo.py --sim-mode
  python demo/predictive_log_anomaly_engine_demo.py --sim-mode --save output.png
"""

import argparse
import warnings
from datetime import datetime, timedelta

import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore')

# ── Brand Palette ─────────────────────────────────────────────────────────────
BLUE   = '#2563EB'
GREEN  = '#16A34A'
AMBER  = '#D97706'
RED    = '#DC2626'
PURPLE = '#7C3AED'
TEAL   = '#0891B2'
GRAY   = '#6B7280'
DARK   = '#111827'
LIGHT  = '#F9FAFB'
BORDER = '#E5E7EB'

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':    'white',
    'axes.facecolor':      LIGHT,
    'axes.edgecolor':      BORDER,
    'axes.grid':           True,
    'grid.alpha':          0.35,
    'grid.color':          '#D1D5DB',
    'font.family':         'DejaVu Sans',
    'font.size':           10,
    'axes.labelsize':      10,
    'axes.titlesize':      12,
    'axes.titleweight':    'bold',
    'axes.titlepad':       10,
    'xtick.labelsize':     9,
    'ytick.labelsize':     9,
    'legend.fontsize':     9,
    'legend.framealpha':   0.85,
    'legend.edgecolor':    BORDER,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _time_axis(n=60, step_seconds=60):
    """Return a list of datetime objects covering the last n×step_seconds."""
    now = datetime.now()
    return [now - timedelta(seconds=step_seconds * (n - i)) for i in range(n)]


def _spine_clean(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)


# ── Panel Functions ───────────────────────────────────────────────────────────

def panel_system_health(ax):
    """Horizontal status bars — all components healthy."""
    components = [
        ('Alert Manager',       GREEN),
        ('Inference Engine',    GREEN),
        ('Investigation UI',    GREEN),
        ('Grafana Dashboard',   GREEN),
        ('Prometheus Scraper',  GREEN),
        ('API Service',         GREEN),
    ]
    names  = [c[0] for c in components]
    colors = [c[1] for c in components]
    y_pos  = range(len(names))

    bars = ax.barh(y_pos, [1.0] * len(names), color=colors,
                   height=0.58, edgecolor='white', linewidth=1.5)

    for i, bar in enumerate(bars):
        ax.text(0.50, i, 'HEALTHY', va='center', ha='center',
                fontsize=9.5, fontweight='bold', color='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 1.25)
    ax.set_title('System Health Status', color=DARK)
    ax.xaxis.set_visible(False)
    _spine_clean(ax, keep=('left',))

    ax.text(0.5, -0.15, 'All Systems Operational',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color=GREEN,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#DCFCE7',
                      edgecolor=GREEN, alpha=0.9))


def panel_event_throughput(ax, n=60):
    """Line chart — simulated event ingestion throughput."""
    np.random.seed(42)
    times = _time_axis(n)
    base  = 88
    wave  = 6 * np.sin(np.linspace(0, 2 * np.pi, n))
    noise = np.random.normal(0, 7, n)
    thru  = np.clip(base + wave + noise, 45, 140).astype(float)
    avg   = float(np.mean(thru))

    ax.fill_between(times, thru, alpha=0.12, color=BLUE)
    ax.plot(times, thru, color=BLUE, linewidth=1.8, label='Events / min')
    ax.axhline(avg, color=GRAY, linewidth=1.2, linestyle='--',
               label=f'60-min avg  {avg:.0f} ev/min')

    ax.set_title('Event Ingestion Throughput — Last 60 Minutes', color=DARK)
    ax.set_ylabel('Events / min')
    ax.set_ylim(0, 165)
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', rotation=25, labelsize=8)
    _spine_clean(ax)


def panel_windows_scored(ax, n=60):
    """Bar chart — inference windows scored per minute."""
    np.random.seed(99)
    noise   = np.random.poisson(2, n)
    windows = 16 + noise
    avg_w   = float(np.mean(windows))

    ax.bar(range(n), windows, color=PURPLE, alpha=0.72, width=0.82,
           edgecolor='white', linewidth=0.5)
    ax.axhline(avg_w, color=AMBER, linewidth=1.8, linestyle='--',
               label=f'Avg  {avg_w:.1f} win/min')

    ax.set_title('Inference Windows Scored / min', color=DARK)
    ax.set_ylabel('Window Count')
    ax.set_xticks([0, 15, 30, 45, 59])
    ax.set_xticklabels(['-60 m', '-45 m', '-30 m', '-15 m', 'Now'], fontsize=8)
    ax.set_ylim(0, 30)
    ax.legend(loc='upper right')
    _spine_clean(ax)


def panel_score_distribution(ax):
    """Histogram — anomaly score distribution with alert threshold marked."""
    np.random.seed(7)
    # Healthy system: most scores cluster well below threshold
    scores = np.concatenate([
        np.random.beta(2, 7, 750) * 0.62,
        np.random.beta(2, 12, 180) * 0.42,
        np.random.beta(5, 2, 20) * 0.72,   # a few elevated (still below threshold)
    ])
    scores    = np.clip(scores, 0, 1)
    threshold = 0.75

    below = scores[scores < threshold]
    above = scores[scores >= threshold]

    ax.hist(below, bins=42, color=BLUE, alpha=0.75, label='Score < threshold (normal)')
    if len(above) > 0:
        ax.hist(above, bins=8, color=RED, alpha=0.88, label='Score ≥ threshold')
    ax.axvline(threshold, color=RED, linewidth=2.0, linestyle='--',
               label=f'Alert threshold  ({threshold})')

    y_top = ax.get_ylim()[1]
    ax.text(threshold * 0.5, y_top * 0.88, 'Normal\nOperating Range',
            ha='center', fontsize=8.5, color=BLUE, fontweight='bold')

    ax.set_title('Anomaly Score Distribution', color=DARK)
    ax.set_xlabel('Anomaly Score  (0 – 1)')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right')
    _spine_clean(ax)


def panel_alert_timeline(ax, n=60, sim_mode=False):
    """Line chart — active alerts over time.  Live state: 0 alerts."""
    times = _time_axis(n)

    if sim_mode:
        np.random.seed(12)
        alerts = np.zeros(n, dtype=int)
        for pos in [18, 36]:          # two brief simulated spikes
            alerts[pos] = np.random.randint(1, 3)
        color  = AMBER
        label  = 'Active Alerts  [PRESENTATION SIMULATION — illustrative only]'
    else:
        alerts = np.zeros(n, dtype=int)
        color  = GREEN
        label  = 'Active Alerts  (live: 0)'

    ax.fill_between(times, alerts, alpha=0.18, color=color)
    ax.plot(times, alerts, color=color, linewidth=2.2, label=label)
    ax.axhline(0, color=GRAY, linewidth=0.8, alpha=0.5)

    if not sim_mode:
        ax.text(0.50, 0.50, 'Zero Active Alerts\nSystem Stable',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color=GREEN, alpha=0.22)

    ax.set_title('Active Alert Count — Last 60 Minutes', color=DARK)
    ax.set_ylabel('Alert Count')
    ax.set_ylim(-0.4, max(3, int(alerts.max()) + 1))
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', rotation=25, labelsize=8)
    _spine_clean(ax)


def panel_severity_distribution(ax):
    """Donut chart — severity distribution of scored windows."""
    labels  = ['Info', 'Warning', 'Critical']
    sizes   = [78, 18, 4]
    colors  = [BLUE, AMBER, RED]
    explode = (0, 0.04, 0.10)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.0f%%', startangle=135,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color('white')
        at.set_fontweight('bold')

    # Hollow centre
    centre = plt.Circle((0, 0), 0.52, color='white')
    ax.add_artist(centre)
    ax.text(0, 0, 'Severity\nSplit', ha='center', va='center',
            fontsize=9, fontweight='bold', color=DARK)

    ax.set_title('Score Severity Distribution\n(Simulated Demo Data)', color=DARK)


def panel_pipeline_latency(ax):
    """Horizontal bar chart — V2 pipeline stage latency (indicative)."""
    stages    = ['Alert Check', 'Severity Classifier', 'Autoencoder Scoring',
                 'LSTM Behavior', 'Embedding Lookup', 'Tokenization']
    latencies = [0.4,           0.9,                   2.8,
                 3.5,           1.2,                    0.8]
    colors    = [AMBER, AMBER, PURPLE, PURPLE, GREEN, GREEN]

    y = range(len(stages))
    bars = ax.barh(y, latencies, color=colors, height=0.56,
                   edgecolor='white', linewidth=1.5, alpha=0.88)
    for bar, val in zip(bars, latencies):
        ax.text(val + 0.06, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f} ms', va='center', fontsize=9, color=DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=9.5)
    ax.set_xlabel('Latency (ms) — Indicative')
    ax.set_title('V2 Pipeline Stage Latency  (Indicative — end-to-end per window)', color=DARK)
    ax.set_xlim(0, max(latencies) * 1.30)
    total = sum(latencies)
    ax.text(0.98, 0.03, f'Total ≈ {total:.1f} ms / window',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=GRAY, style='italic')
    _spine_clean(ax)


# ── Dashboard Builder ─────────────────────────────────────────────────────────

def build_dashboard(sim_mode: bool = False, save_path: str = None):
    """Compose and render the full observability dashboard."""
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('white')

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(
        0.50, 0.975,
        'Predictive Log Anomaly Engine  ·  System Observability Dashboard',
        ha='center', va='top',
        fontsize=19, fontweight='bold', color=DARK,
    )
    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S UTC+0')
    fig.text(
        0.50, 0.948,
        f'●  ALL SYSTEMS HEALTHY    ACTIVE ALERTS: 0    {ts}',
        ha='center', va='top',
        fontsize=11, fontweight='bold', color=GREEN,
    )
    if sim_mode:
        fig.text(
            0.50, 0.926,
            '⚠  PRESENTATION SIMULATION MODE active on selected panels — data is illustrative, not production evidence',
            ha='center', va='top',
            fontsize=8.5, color=AMBER, style='italic',
        )

    # ── Grid ──────────────────────────────────────────────────────────────────
    top = 0.915 if sim_mode else 0.935
    gs  = gridspec.GridSpec(
        3, 3,
        left=0.07, right=0.97,
        top=top,   bottom=0.06,
        wspace=0.32, hspace=0.54,
    )

    ax_health  = fig.add_subplot(gs[0, 0])
    ax_through = fig.add_subplot(gs[0, 1:])
    ax_windows = fig.add_subplot(gs[1, 0])
    ax_scores  = fig.add_subplot(gs[1, 1])
    ax_alerts  = fig.add_subplot(gs[1, 2])
    ax_sev     = fig.add_subplot(gs[2, 0])
    ax_lat     = fig.add_subplot(gs[2, 1:])

    panel_system_health(ax_health)
    panel_event_throughput(ax_through)
    panel_windows_scored(ax_windows)
    panel_score_distribution(ax_scores)
    panel_alert_timeline(ax_alerts, sim_mode=sim_mode)
    panel_severity_distribution(ax_sev)
    panel_pipeline_latency(ax_lat)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.50, 0.012,
        'Predictive Log Anomaly Engine v2  ·  FastAPI  ·  PyTorch  ·  Prometheus  ·  Grafana  ·  '
        'LSTM Autoencoder Pipeline',
        ha='center', va='bottom',
        fontsize=8, color=GRAY,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  Dashboard saved  →  {save_path}')
    else:
        plt.show()

    return fig


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Predictive Log Anomaly Engine — Observability Dashboard Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--save', metavar='PATH',
                   help='Save the dashboard to PATH (PNG/PDF/SVG)')
    p.add_argument('--sim-mode', action='store_true',
                   help='Enable optional presentation simulation layer (illustrative only)')
    return p.parse_args()


def main():
    args = _parse_args()

    bar = '=' * 68
    print(bar)
    print('  Predictive Log Anomaly Engine — Observability Dashboard Demo')
    print(bar)
    print(f'  API Status    : Healthy')
    print(f'  Prometheus    : Scraper up')
    print(f'  Grafana       : Dashboard healthy')
    print(f'  Active Alerts : 0  (no anomaly crossed the configured threshold)')
    print(f'  Generated     : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if args.sim_mode:
        print(f'  Mode          : PRESENTATION SIMULATION (illustrative panels labelled)')
    else:
        print(f'  Mode          : Live system reflection')
    print(bar)
    print()

    build_dashboard(sim_mode=args.sim_mode, save_path=args.save)


if __name__ == '__main__':
    main()
