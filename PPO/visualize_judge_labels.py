import json
import matplotlib.pyplot as plt
from collections import Counter

# Suffixes to process
SUFFIXES = ["sft", "sft_ppo"]

def parse_label(raw):
    """
    Robustly parse a judge_label that might look like:
      "2", "2\n", "2 (implicitly affirms)", etc.
    Returns an int 0–3 or None if it can't parse.
    """
    if raw is None:
        return None
    s = str(raw).strip()

    # If the whole thing is just a digit, easy
    if s.isdigit():
        val = int(s)
        return val if 0 <= val <= 3 else None

    # Otherwise, grab the first digit we see
    for ch in s:
        if ch.isdigit():
            val = int(ch)
            return val if 0 <= val <= 3 else None

    return None


for suffix in SUFFIXES:
    INPUT_JSON = f"oem_val_judged_{suffix}.json"
    OUTPUT_IMAGE = f"judge_label_distribution_{suffix}.jpg"
    
    print(f"\n{'='*60}")
    print(f"Processing: {suffix}")
    print(f"Input: {INPUT_JSON}")
    print(f"Output: {OUTPUT_IMAGE}")
    print(f"{'='*60}\n")

    # Load JSON data
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # Count judge labels
    label_counts = Counter()
    for entry in data:
        raw_label = entry.get("judge_label", None)
        label = parse_label(raw_label)
        if label is not None:
            label_counts[label] += 1

    # Prepare data for plotting
    labels = [0, 1, 2, 3]
    counts = [label_counts.get(label, 0) for label in labels]

    # Determine title based on suffix
    title = "Distribution of Judge Labels"
    if suffix == "human":
        title += " - Human Responses"
    elif suffix == "sft":
        title += " - SFT Model"
    elif suffix == "baseline":
        title += " - Baseline Model"

    # Create bar graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, edgecolor="black", linewidth=1.2)

    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Customize plot
    plt.xlabel("Judge Label", fontsize=14, fontweight="bold")
    plt.ylabel("Frequency", fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(
        labels,
        [
            "0\n(Challenges\nActions)",
            "1\n(Neutral/\nUnrelated)",
            "2\n(Implicit\nAffirmation)",
            "3\n(Explicit\nAffirmation)",
        ],
    )
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Save figure
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    print(f"Bar graph saved to: {OUTPUT_IMAGE}")

    # Print summary statistics
    total = sum(counts)
    print(f"\nTotal entries: {total}")
    print("Label distribution:")
    for label, count in zip(labels, counts):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  Label {label}: {count} ({percentage:.1f}%)")
    
    plt.close()

print(f"\n{'='*60}")
print("All visualizations created successfully!")
print(f"{'='*60}")