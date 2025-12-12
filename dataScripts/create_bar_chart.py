"""
create_bar_chart.py

Creates a grouped bar chart comparing judge label distributions across multiple models.
Each model's responses are evaluated for sycophancy levels (0-3) and displayed side-by-side.

Usage:
    python create_bar_chart.py
"""

import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# ============================================================
# USER CONFIGURATION
# ============================================================

# List of files to compare
# Each entry contains:
#   - filename: path to JSON file with judge labels
#   - label_key: field name containing the judge label
#   - bar_label: display name for this model in the legend
FILES_TO_PROCESS = [
    {
        "filename": "oem_val_judged_human.json",
        "label_key": "judge_label_human",
        "bar_label": "Human",
    },
    {
        "filename": "oem_val_judged_baseline.json",
        "label_key": "judge_label_base_reply",
        "bar_label": "Baseline",
    },
    {
        "filename": "oem_val_judged_sft.json",
        "label_key": "judge_label_sft_reply",
        "bar_label": "SFT",
    },
    {
        "filename": "oem_val_judged_sft_ppo.json",
        "label_key": "judge_label_sft_ppo_reply",
        "bar_label": "SFT + PPO",
    },
]

# Output image settings
OUTPUT_IMAGE = "judge_label_distribution_comparison.jpg"
OUTPUT_DPI = 300
FIGURE_SIZE = (12, 7)

# Chart appearance settings
CHART_TITLE = "Distribution of Judge Labels Across Models"
X_AXIS_LABEL = "Judge Label"
Y_AXIS_LABEL = "Frequency"
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1.3

# Font sizes
TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 13
BAR_COUNT_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
LEGEND_TITLE_FONT_SIZE = 15

# ============================================================
# SCRIPT LOGIC - DO NOT MODIFY BELOW
# ============================================================

def parse_label(raw):
    """
    Robustly parse judge_label variants.
    Returns int in [0,3] or None.
    """
    if raw is None:
        return None
    s = str(raw).strip()

    # Exact digit
    if s.isdigit():
        val = int(s)
        return val if 0 <= val <= 3 else None

    # Otherwise scan first digit
    for ch in s:
        if ch.isdigit():
            val = int(ch)
            return val if 0 <= val <= 3 else None

    return None

def load_and_count_labels(filename, label_key):
    """
    Load JSON file and count occurrences of each judge label (0-3).
    Returns: (counts_array, total_count)
    """
    with open(filename, "r") as f:
        data = json.load(f)
    
    label_counts = Counter()
    for entry in data:
        raw_label = entry.get(label_key, None)
        label = parse_label(raw_label)
        if label is not None:
            label_counts[label] += 1
    
    # Convert to array [count_0, count_1, count_2, count_3]
    labels = [0, 1, 2, 3]
    counts = [label_counts.get(l, 0) for l in labels]
    total = sum(counts)
    
    return counts, total

def create_bar_chart():
    """
    Create grouped bar chart comparing judge label distributions.
    """
    print(f"\n{'='*60}")
    print(f"Creating bar chart with {len(FILES_TO_PROCESS)} models")
    print(f"Output: {OUTPUT_IMAGE}")
    print(f"{'='*60}\n")
    
    # Labels for x-axis
    labels = [0, 1, 2, 3]
    label_descriptions = [
        "0\n(Challenges\nActions)",
        "1\n(Neutral/\nUnrelated)",
        "2\n(Implicit\nAffirmation)",
        "3\n(Explicit\nAffirmation)"
    ]
    
    # Load data for each model
    model_counts = []
    model_totals = []
    
    for cfg in FILES_TO_PROCESS:
        filename = cfg["filename"]
        label_key = cfg["label_key"]
        bar_label = cfg["bar_label"]
        
        print(f"Processing: {bar_label}")
        print(f"  File: {filename}")
        print(f"  Label key: {label_key}")
        
        try:
            counts, total = load_and_count_labels(filename, label_key)
            model_counts.append(counts)
            model_totals.append(total)
            
            print(f"  Total entries: {total}")
            print(f"  Distribution: {counts}")
            for lbl, c in zip(labels, counts):
                pct = (c / total * 100) if total > 0 else 0.0
                print(f"    Label {lbl}: {c} ({pct:.1f}%)")
            print()
            
        except FileNotFoundError:
            print(f"  ERROR: File not found!")
            print(f"  Skipping this model...\n")
            model_counts.append([0, 0, 0, 0])
            model_totals.append(0)
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping this model...\n")
            model_counts.append([0, 0, 0, 0])
            model_totals.append(0)
    
    # Create plot
    print("Generating chart...")
    num_labels = len(labels)
    num_models = len(FILES_TO_PROCESS)
    
    x = np.arange(num_labels)  # Base x positions: 0,1,2,3
    bar_width = 0.8 / num_models
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot bars for each model
    for i, cfg in enumerate(FILES_TO_PROCESS):
        bar_label = cfg["bar_label"]
        counts = model_counts[i]
        
        # Calculate x positions for this model's bars
        offsets = x + (i - (num_models - 1) / 2) * bar_width
        
        # Create bars
        bars = plt.bar(
            offsets,
            counts,
            width=bar_width,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=bar_label,
        )
        
        # Add count labels above bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{count}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_COUNT_FONT_SIZE,
                    fontweight="bold"
                )
    
    # Customize plot
    plt.xticks(x, label_descriptions, fontsize=TICK_LABEL_FONT_SIZE)
    plt.xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
    plt.ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
    plt.title(CHART_TITLE, fontsize=TITLE_FONT_SIZE, fontweight="bold")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.legend(
        title="Model",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"✓ Chart saved to: {OUTPUT_IMAGE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    create_bar_chart()
