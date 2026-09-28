"""Epoch history and vector PDF learning curves for training entry points."""
import csv
import math
import os
from pathlib import Path


def record_epoch(history, epoch, phase, scores):
    """Merge hooks regardless of whether validation runs before training ends."""
    row = next((r for r in history if r["epoch"] == int(epoch)), None)
    if row is None:
        row = {"epoch": int(epoch)}
        history.append(row)
    row.update({f"{phase}_loss": float(scores["loss"]),
                f"{phase}_pcc": float(scores["pcc"])})


def save_training_curves(history, output_dir):
    """Write CSV and two vector PDFs. Call on global rank zero only."""
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(history, key=lambda row: row["epoch"])
    fields = ["epoch", "train_loss", "val_loss", "train_pcc", "val_pcc"]
    with (directory / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)
    with plt.rc_context({"pdf.fonttype": 42, "font.size": 11}):
        for metric, label in (("loss", "Loss"), ("pcc", "Pearson correlation (PCC)")):
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            try:
                for phase, name, color in (("train", "Training", "#2166ac"),
                                           ("val", "Validation", "#b2182b")):
                    key = f"{phase}_{metric}"
                    values = [(r["epoch"] + 1, r[key]) for r in rows
                              if key in r and math.isfinite(r[key])]
                    if values:
                        x, y = zip(*values)
                        ax.plot(x, y, label=name, color=color, marker="o", markersize=3,
                                linewidth=1.5, rasterized=False)
                ax.set(xlabel="Epoch", ylabel=label, title=f"Training and validation {metric.upper() if metric == 'pcc' else metric}")
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(alpha=.2)
                if ax.lines: ax.legend(frameon=False)
                fig.tight_layout()
                path = directory / f"training_validation_{metric}.pdf"
                temporary = path.with_suffix(".pdf.tmp")
                fig.savefig(temporary, format="pdf", bbox_inches="tight")
                os.replace(temporary, path)
            finally:
                plt.close(fig)
