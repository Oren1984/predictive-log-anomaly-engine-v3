#!/usr/bin/env python3
"""
Predictive Log Anomaly Engine — GPU Execution Path Demo
========================================================

Presentation-grade, standalone demo script visualising the optional GPU
acceleration path of the Predictive Log Anomaly Engine.

The standard CPU pipeline is FULLY PRODUCTION-READY for typical log volumes.
GPU is an additive capability — same PyTorch models, same architecture, same
training artifacts — targeted at CUDA devices for higher throughput at scale.

ALL PERFORMANCE FIGURES IN THIS SCRIPT ARE CONCEPTUAL / ILLUSTRATIVE.
They are not derived from production measurements.  Real numbers depend on
hardware, batch size, log volume, and model configuration.

Usage
------
  python demo/predictive_log_anomaly_engine_gpu_demo.py
  python demo/predictive_log_anomaly_engine_gpu_demo.py --save gpu_demo.png
"""

import argparse
import warnings
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore')

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE       = '#2563EB'
GREEN      = '#16A34A'
AMBER      = '#D97706'
RED        = '#DC2626'
PURPLE     = '#7C3AED'
GRAY       = '#6B7280'
DARK       = '#111827'
LIGHT      = '#F9FAFB'
BORDER     = '#E5E7EB'
GPU_COLOR  = '#22C55E'   # vibrant green — GPU path
CPU_COLOR  = '#3B82F6'   # blue — CPU path

plt.rcParams.update({
    'figure.facecolor':   'white',
    'axes.facecolor':     LIGHT,
    'axes.edgecolor':     BORDER,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'grid.color':         '#D1D5DB',
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.labelsize':     10,
    'axes.titlesize':     12,
    'axes.titleweight':   'bold',
    'axes.titlepad':      10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   BORDER,
})


def _spine_clean(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)


# ── Panel Functions ───────────────────────────────────────────────────────────

def panel_execution_path(ax):
    """Side-by-side CPU vs GPU pipeline flow diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis('off')
    ax.set_facecolor(LIGHT)
    ax.set_title('Execution Path: CPU Runtime vs. GPU Runtime (Optional)', color=DARK, pad=12)

    # Column headers
    ax.text(2.5, 6.9, 'Standard CPU Runtime', ha='center', fontsize=11,
            fontweight='bold', color=CPU_COLOR)
    ax.text(7.5, 6.9, 'GPU Runtime  (Optional Path)', ha='center', fontsize=11,
            fontweight='bold', color=GPU_COLOR)
    ax.axvline(5.0, color=BORDER, linewidth=1.5, ymin=0.06, ymax=0.96)

    cpu_stages = [
        ('Log Tokenization',   False),
        ('Word2Vec Lookup',     False),
        ('LSTM Inference',      False),
        ('Autoencoder Scoring', False),
        ('Alert Check',         False),
    ]
    gpu_stages = [
        ('Log Tokenization',    False),
        ('Word2Vec Lookup',     False),
        ('LSTM  (CUDA device)', True),
        ('Autoencoder (CUDA)',  True),
        ('Alert Check',         False),
    ]

    for i, (label, _accel) in enumerate(cpu_stages):
        y = 5.9 - i * 1.05
        r = FancyBboxPatch((0.4, y - 0.30), 4.2, 0.60,
                           boxstyle='round,pad=0.06',
                           facecolor=CPU_COLOR, edgecolor='white',
                           linewidth=1.8, alpha=0.85)
        ax.add_patch(r)
        ax.text(2.5, y, label, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        if i < len(cpu_stages) - 1:
            ax.annotate('', xy=(2.5, y - 0.30), xytext=(2.5, y - 0.75),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.4))

    for i, (label, accel) in enumerate(gpu_stages):
        y     = 5.9 - i * 1.05
        color = GPU_COLOR if accel else CPU_COLOR
        r = FancyBboxPatch((5.4, y - 0.30), 4.2, 0.60,
                           boxstyle='round,pad=0.06',
                           facecolor=color, edgecolor='white',
                           linewidth=1.8, alpha=0.85)
        ax.add_patch(r)
        suffix = '  ⚡' if accel else ''
        ax.text(7.5, y, label + suffix, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        if i < len(gpu_stages) - 1:
            ax.annotate('', xy=(7.5, y - 0.30), xytext=(7.5, y - 0.75),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.4))

    # Bottom badges
    ax.text(2.5, 0.25, '✓  Fully supported · All hardware\nRecommended default',
            ha='center', fontsize=8.5, color=CPU_COLOR, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#DBEAFE',
                      edgecolor=CPU_COLOR, alpha=0.8))
    ax.text(7.5, 0.25, '⚡  Requires CUDA-capable GPU\nOptional — additive, not mandatory',
            ha='center', fontsize=8.5, color=GPU_COLOR, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#DCFCE7',
                      edgecolor=GPU_COLOR, alpha=0.8))


def panel_throughput_scaling(ax):
    """Conceptual throughput vs. batch size for CPU and GPU paths."""
    batch_sizes    = [1, 4, 8, 16, 32, 64, 128, 256]
    cpu_throughput = [100, 175, 235, 280, 310, 325, 332, 335]
    gpu_throughput = [88,  195, 380, 620, 940, 1380, 1880, 2350]

    ax.plot(batch_sizes, cpu_throughput, 'o-', color=CPU_COLOR, linewidth=2.5,
            markersize=7, label='CPU (Standard Runtime)',
            markerfacecolor='white', markeredgewidth=2.2)
    ax.plot(batch_sizes, gpu_throughput, 's-', color=GPU_COLOR, linewidth=2.5,
            markersize=7, label='GPU (CUDA Runtime) — Conceptual',
            markerfacecolor='white', markeredgewidth=2.2)
    ax.fill_between(batch_sizes, cpu_throughput, alpha=0.08, color=CPU_COLOR)
    ax.fill_between(batch_sizes, gpu_throughput, alpha=0.08, color=GPU_COLOR)

    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel('Batch Size (log₂ scale)')
    ax.set_ylabel('Throughput (windows / sec)  — Illustrative')
    ax.set_title('Throughput vs. Batch Size\n'
                 '[Conceptual Illustration — Not Measured Benchmarks]', color=DARK)
    ax.legend(loc='upper left')

    ax.text(0.98, 0.06,
            'All figures are conceptual illustrations of\n'
            'general GPU scaling behaviour.\n'
            'Not derived from production measurements.',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color=GRAY, style='italic',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow',
                      edgecolor=AMBER, alpha=0.85))
    _spine_clean(ax)


def panel_gpu_applicability(ax):
    """Stacked horizontal bar — GPU applicability per pipeline stage."""
    stages    = ['Alert Manager', 'Severity Classifier',
                 'Autoencoder Scoring', 'LSTM Behavior Model',
                 'Embedding Lookup', 'Tokenization']
    cpu_frac  = [100, 75,  55,  60,  100, 100]   # % handled on CPU regardless
    gpu_frac  = [0,   25,  45,  40,  0,   0]     # % potential GPU acceleration

    y = range(len(stages))
    ax.barh(y, cpu_frac, color=CPU_COLOR, height=0.56,
            edgecolor='white', linewidth=1.5, alpha=0.80, label='CPU execution')
    ax.barh(y, gpu_frac, left=cpu_frac, color=GPU_COLOR, height=0.56,
            edgecolor='white', linewidth=1.5, alpha=0.88,
            label='GPU acceleration potential ⚡  (illustrative)')

    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=9.5)
    ax.set_xlabel('Execution Profile  (%)')
    ax.set_title('GPU Applicability per Pipeline Stage\n(Illustrative)', color=DARK)
    ax.legend(loc='lower right', fontsize=8.5)
    ax.set_xlim(0, 125)
    _spine_clean(ax)


def panel_deployment_decision(ax):
    """2×2 deployment decision positioning matrix (text-based)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor(LIGHT)
    ax.set_title('Deployment Decision Framework', color=DARK, pad=12)

    # Axis labels
    ax.text(5.0, 7.65, 'Log Volume  →', ha='center', fontsize=9.5, color=GRAY)
    ax.text(0.35, 4.0, 'Latency\nRequirement  ↑',
            ha='center', va='center', fontsize=9.5, color=GRAY, rotation=90)
    ax.axvline(5.0, color=BORDER, linewidth=1.5, ymin=0.08, ymax=0.90)
    ax.axhline(4.0, color=BORDER, linewidth=1.5, xmin=0.06, xmax=0.97)

    quadrants = [
        (2.8, 5.8, 'Standard CPU\n✓  Recommended', CPU_COLOR,
         'Low–Medium volume\nRelaxed latency SLA', '#DBEAFE'),
        (7.2, 5.8, 'GPU Optional\n⚡  Consider', GPU_COLOR,
         'High volume\nRelaxed latency SLA', '#DCFCE7'),
        (2.8, 2.2, 'Standard CPU\n✓  Recommended', CPU_COLOR,
         'Low–Medium volume\nStrict latency SLA', '#DBEAFE'),
        (7.2, 2.2, 'GPU Recommended\n⚡  Evaluate', AMBER,
         'High volume\nStrict latency SLA', '#FEF9C3'),
    ]

    for x, y, title, color, subtitle, bg in quadrants:
        ax.add_patch(FancyBboxPatch((x - 1.9, y - 1.3), 3.8, 2.6,
                                   boxstyle='round,pad=0.1',
                                   facecolor=bg, edgecolor=color,
                                   linewidth=1.5, alpha=0.65))
        ax.text(x, y + 0.55, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
        ax.text(x, y - 0.45, subtitle, ha='center', va='center',
                fontsize=8, color=GRAY)

    ax.text(5.0, 0.35,
            'Current system is well-served by the Standard CPU runtime.',
            ha='center', fontsize=8.5, color=DARK, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#DCFCE7',
                      edgecolor=GREEN, alpha=0.85))


def panel_model_overview(ax):
    """Grouped bar — model sizes and conceptual GPU speedup potential."""
    models    = ['Word2Vec\nEmbeddings', 'LSTM\nBehavior', 'Autoencoder\nDetector',
                 'Severity\nClassifier']
    params_m  = [0.5, 2.1, 1.8, 0.9]        # indicative parameter counts (M)
    gpu_mult  = [1.0, 3.5, 3.0, 1.5]        # conceptual batch speedup potential

    x     = np.arange(len(models))
    width = 0.36

    bars1 = ax.bar(x - width / 2, params_m, width, label='Parameters (M) — Indicative',
                   color=PURPLE, edgecolor='white', linewidth=1.5, alpha=0.85)

    ax2   = ax.twinx()
    bars2 = ax2.bar(x + width / 2, gpu_mult, width,
                    label='GPU Speedup Potential (×) — Conceptual',
                    color=GPU_COLOR, edgecolor='white', linewidth=1.5, alpha=0.78)
    ax2.set_ylabel('Conceptual GPU Speedup (×)', color=GPU_COLOR)
    ax2.tick_params(axis='y', labelcolor=GPU_COLOR)
    ax2.set_ylim(0, 5.5)
    ax2.axhline(1.0, color=CPU_COLOR, linewidth=1.2, linestyle=':', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Parameters (Millions) — Indicative', color=PURPLE)
    ax.tick_params(axis='y', labelcolor=PURPLE)
    ax.set_ylim(0, 3.5)
    ax.set_title('Model Complexity & GPU Potential\n'
                 '(Indicative — Not Measured)', color=DARK)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc='upper right', fontsize=8)
    _spine_clean(ax)


def panel_runtime_notes(ax):
    """Key runtime and tradeoff notes as a structured text panel."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    ax.set_facecolor(LIGHT)
    ax.set_title('Runtime Considerations & Tradeoffs', color=DARK, pad=12)

    notes = [
        ('✓  CPU (Standard)',      GPU_COLOR if False else CPU_COLOR,
         'Default path — runs on any hardware, fully supported.'),
        ('✓  CUDA GPU',            GPU_COLOR,
         'Same PyTorch models — target via torch.device("cuda").'),
        ('⚡  GPU ideal for',       GPU_COLOR,
         'Large batch scoring, high-freq streaming, offline bulk inference.'),
        ('—  GPU overkill for',    GRAY,
         'Single-node moderate volume — standard CPU is perfectly adequate.'),
        ('→  Portability',         BLUE,
         'No architecture changes needed; set device in runtime config.'),
        ('→  Deployment',          BLUE,
         'Set CUDA_VISIBLE_DEVICES or pass torch.device to model.to().'),
        ('⚠  Overhead note',       AMBER,
         'At batch_size=1, GPU may be slower than CPU (transfer cost).'),
    ]

    for i, (label, color, detail) in enumerate(notes):
        y = 7.9 - i * 1.08
        ax.text(0.3, y, label, fontsize=9.5, fontweight='bold', color=color)
        ax.text(0.3, y - 0.44, detail, fontsize=8.5, color=DARK)
        ax.axhline(y - 0.76, xmin=0.02, xmax=0.98,
                   color=BORDER, linewidth=0.9)


# ── Dashboard Builder ─────────────────────────────────────────────────────────

def build_gpu_dashboard(save_path: str = None):
    """Compose and render the GPU demo dashboard."""
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('white')

    # Header
    fig.text(0.50, 0.975,
             'Predictive Log Anomaly Engine  ·  GPU Execution Path Overview',
             ha='center', va='top',
             fontsize=19, fontweight='bold', color=DARK)
    fig.text(0.50, 0.948,
             'Optional Advanced Runtime Acceleration Path  ·  '
             'All performance figures are conceptual / illustrative',
             ha='center', va='top',
             fontsize=11, fontweight='bold', color=GPU_COLOR)
    fig.text(0.50, 0.924,
             'The standard CPU pipeline is fully production-ready — '
             'GPU adds throughput at scale, not correctness',
             ha='center', va='top',
             fontsize=9, color=GRAY, style='italic')

    gs = gridspec.GridSpec(
        3, 3,
        left=0.06, right=0.97,
        top=0.915, bottom=0.06,
        wspace=0.34, hspace=0.58,
    )

    ax_path   = fig.add_subplot(gs[0, :2])
    ax_notes  = fig.add_subplot(gs[0, 2])
    ax_batch  = fig.add_subplot(gs[1, :2])
    ax_deploy = fig.add_subplot(gs[1, 2])
    ax_gpu    = fig.add_subplot(gs[2, :2])
    ax_model  = fig.add_subplot(gs[2, 2])

    panel_execution_path(ax_path)
    panel_runtime_notes(ax_notes)
    panel_throughput_scaling(ax_batch)
    panel_deployment_decision(ax_deploy)
    panel_gpu_applicability(ax_gpu)
    panel_model_overview(ax_model)

    # Footer
    fig.text(
        0.50, 0.012,
        'Predictive Log Anomaly Engine v2  ·  PyTorch CUDA-compatible  ·  '
        'LSTM + Autoencoder  ·  Optional GPU Acceleration Path',
        ha='center', va='bottom',
        fontsize=8, color=GRAY,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'  GPU dashboard saved  →  {save_path}')
    else:
        plt.show()

    return fig


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Predictive Log Anomaly Engine — GPU Execution Path Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--save', metavar='PATH',
                   help='Save dashboard to PATH (PNG/PDF/SVG)')
    return p.parse_args()


def main():
    args = _parse_args()

    bar = '=' * 68
    print(bar)
    print('  Predictive Log Anomaly Engine — GPU Execution Path Demo')
    print(bar)
    print(f'  GPU Mode   : Optional advanced acceleration (CUDA-compatible)')
    print(f'  CPU Mode   : Recommended default — fully production-ready')
    print(f'  Figures    : All performance data is conceptual / illustrative')
    print(f'  Generated  : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(bar)
    print()

    build_gpu_dashboard(save_path=args.save)


if __name__ == '__main__':
    main()
