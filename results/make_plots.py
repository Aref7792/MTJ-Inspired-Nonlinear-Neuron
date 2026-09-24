"""Parse training logs and save plots as PNG images.

Usage: python results/make_plots.py [date]   (default: today)
"""

import os
import re
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")

MODELS = [
    ("blif", "BLIF (Leaky LIF baseline)", "#4C78A8"),
    ("smtj", "SMTJ (soft MTJ)", "#F58518"),
    ("hmtj", "HMTJ (hard MTJ)", "#54A24B"),
    ("thmtj", "Ternary MTJ", "#E45756"),
]


def parse(path):
    """Return per-epoch metrics and the final summary from one run log."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    epochs = []
    blocks = re.split(r"^Epoch (\d+)/\d+$", text, flags=re.M)

    # blocks = [preamble, "1", body1, "2", body2, ...]
    for i in range(1, len(blocks) - 1, 2):
        n = int(blocks[i])
        body = blocks[i + 1]

        train = re.search(
            r"-+ TRAIN -+\s*\nLoss:\s+([\d.]+)\s*\nAccuracy:\s+([\d.]+)%"
            r".*?Layer 1:\s+([\d.]+)\s*\nLayer 2:\s+([\d.]+)\s*\nLayer 3:\s+([\d.]+)",
            body,
            re.S,
        )
        test = re.search(
            r"-+ TEST -+\s*\nAccuracy:\s+([\d.]+)%"
            r".*?Layer 1:\s+([\d.]+)\s*\nLayer 2:\s+([\d.]+)\s*\nLayer 3:\s+([\d.]+)",
            body,
            re.S,
        )
        if not (train and test):
            continue

        epochs.append(
            {
                "epoch": n,
                "train_loss": float(train.group(1)),
                "train_acc": float(train.group(2)),
                "test_acc": float(test.group(1)),
                "fr1": float(test.group(2)),
                "fr2": float(test.group(3)),
                "fr3": float(test.group(4)),
            }
        )

    summary = {}
    for key, pat in (
        ("best", r"Best Test Accuracy:\s+([\d.]+)%"),
        ("final", r"Final Test Accuracy:\s+([\d.]+)%"),
        ("spike_count", r"Spike-count Test Accuracy:\s+([\d.]+)%"),
    ):
        m = re.search(pat, text)
        if m:
            summary[key] = float(m.group(1))

    return epochs, summary


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    stamp = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    os.makedirs(OUT, exist_ok=True)

    runs = {}
    for key, label, color in MODELS:
        path = os.path.join(HERE, f"{key}_cpu_run_{stamp}.log")
        if not os.path.exists(path):
            print("missing", path)
            continue
        epochs, summary = parse(path)
        if not epochs:
            print("no complete epochs yet in", path)
            continue
        runs[key] = (label, color, epochs, summary)

    if not runs:
        print("nothing to plot")
        return

    # ---- per-model figure: loss, accuracy, firing rates -------------------
    for key, (label, color, ep, summary) in runs.items():
        x = [e["epoch"] for e in ep]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        ax[0].plot(x, [e["train_loss"] for e in ep], "o-", color=color)
        ax[0].set_title("Training loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        ax[1].plot(x, [e["train_acc"] for e in ep], "o-", label="train")
        ax[1].plot(x, [e["test_acc"] for e in ep], "s-", label="test")
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy (%)")
        ax[1].legend()

        for i, name in enumerate(("fr1", "fr2", "fr3"), start=1):
            ax[2].plot(x, [e[name] for e in ep], "o-", label=f"Layer {i}")
        ax[2].set_title("Test firing rates")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("Firing rate")
        ax[2].legend()

        for a in ax:
            a.grid(alpha=0.3)

        fig.suptitle(label)
        save(fig, f"{key}_{stamp}.png")

    # ---- comparison across models ----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, (label, color, ep, _) in runs.items():
        x = [e["epoch"] for e in ep]
        ax[0].plot(x, [e["test_acc"] for e in ep], "o-", color=color, label=label)
        ax[1].plot(x, [e["train_loss"] for e in ep], "o-", color=color, label=label)

    ax[0].set_title("Test accuracy")
    ax[0].set_ylabel("Accuracy (%)")
    ax[1].set_title("Training loss")
    ax[1].set_ylabel("Loss")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)

    fig.suptitle(f"MNIST SNN comparison — CPU run {stamp}")
    save(fig, f"comparison_{stamp}.png")

    # ---- best-accuracy bar chart -----------------------------------------
    done = {k: v for k, v in runs.items() if "best" in v[3]}
    if done:
        fig, a = plt.subplots(figsize=(6, 4))
        labels = [runs[k][0].split(" (")[0] for k in done]
        vals = [runs[k][3]["best"] for k in done]
        bars = a.bar(labels, vals, color=[runs[k][1] for k in done])
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom")
        a.set_ylabel("Best test accuracy (%)")
        a.set_ylim(min(vals) - 2, 100)
        a.grid(axis="y", alpha=0.3)
        a.set_title(f"Best test accuracy — CPU run {stamp}")
        save(fig, f"best_accuracy_{stamp}.png")

    # ---- summary table to stdout -----------------------------------------
    print("\n| Model | Epochs | Best test | Final test | Spike-count |")
    print("|---|---:|---:|---:|---:|")
    for key, (label, _, ep, s) in runs.items():
        print(
            f"| {label} | {len(ep)} | "
            f"{s.get('best', float('nan')):.2f}% | "
            f"{s.get('final', float('nan')):.2f}% | "
            f"{s.get('spike_count', float('nan')):.2f}% |"
        )


if __name__ == "__main__":
    main()
