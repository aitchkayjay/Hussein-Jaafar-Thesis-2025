# plot_kitti_loss.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv_path",
        nargs="?",
        default="/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mixed/loss_history.csv",
        help="Pfad zur loss_history.csv"
    )
    ap.add_argument("--title", default="KITTI+CARLA training loss", help="Plot Titel")
    ap.add_argument("--out", default=None, help="optional eigener Ausgabepfad für die Grafik")
    args = ap.parse_args()

    csv = Path(args.csv_path)
    df = pd.read_csv(csv)

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    loss_col = "train_loss" if "train_loss" in df.columns else "loss"
    df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")
    df = df.dropna(subset=["epoch", loss_col]).sort_values("epoch")

    x = df["epoch"].to_numpy()
    y = df[loss_col].to_numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title(args.title)
    plt.grid(True)

    i_min = int(np.argmin(y))
    x_min = int(x[i_min])
    y_min = float(y[i_min])
    plt.scatter([x_min], [y_min])
    plt.annotate(
        f"min {y_min:.4f} @ {x_min}",
        xy=(x_min, y_min),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom"
    )

    out_png = Path(args.out) if args.out else csv.with_name(csv.stem + "_loss.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"gespeichert {out_png}")

if __name__ == "__main__":
    main()
