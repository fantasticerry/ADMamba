import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    labels = ["2_dir", "4_dir", "8_dir", "8_sel_4"]
    values = [0.652, 0.636, 0.633, 0.587]
    colors = ["#F5F0D0", "#B8CC6A", "#8DB63C", "#2E4E2E"]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    bars = ax.bar(labels, values, width=0.52, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold",
                fontfamily="serif")

    ax.set_ylabel("Cosine Similarity", fontsize=16, fontweight="bold",
                  labelpad=8)
    ax.set_xlabel("")

    ax.set_ylim(0.56, 0.68)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))

    ax.tick_params(axis="both", which="major", labelsize=13, width=1.0,
                   length=4, direction="out")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")

    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig("cosine_similarity_bar.png", dpi=300, bbox_inches="tight",
                pad_inches=0.08)
    plt.savefig("cosine_similarity_bar.pdf", bbox_inches="tight",
                pad_inches=0.08)
    print("Saved → cosine_similarity_bar.png / .pdf")


if __name__ == "__main__":
    main()
