import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import load_results_json, fano_upper_accuracy_from_I


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    res = load_results_json(args.input_json)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    labels = res["labels"]
    x = np.arange(len(labels))

    accs = np.array(res["accs"])
    ixz = np.array(res["ixz"])
    izy = np.array(res["izy"])
    compr = np.array(res["compr"])

    theo_acc = fano_upper_accuracy_from_I(
        izy,
        K=10,
        H_Y_bits=math.log2(10),
    )
    theo_acc = 100.0 * np.array(theo_acc)

    plt.figure(figsize=(12, 6))
    plt.plot(x, accs, marker="o")
    plt.xlabel("config (fc1|fc2|fc3)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_config.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(x, ixz, marker="o")
    plt.xlabel("config (fc1|fc2|fc3)")
    plt.ylabel("I(X;Z) (bits)")
    plt.title("I(X;Z) vs Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ixz_vs_config.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(x, izy, marker="o")
    plt.xlabel("config (fc1|fc2|fc3)")
    plt.ylabel("I(Z;Y) (bits)")
    plt.title("I(Z;Y) vs Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "izy_vs_config.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        x,
        100.0 * accs,
        marker="o",
        label="Actual Accuracy (%)",
    )
    plt.plot(
        x,
        theo_acc,
        marker="s",
        label="Theoretical (Fano upper bound) (%)",
    )
    plt.xlabel("config (fc1|fc2|fc3)")
    plt.ylabel("Accuracy (%)")
    plt.title("Actual vs Theoretical Accuracy")
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_theoretical.png"))
    plt.close()

    err_counts = np.array(res["err_counts"])
    err_shares = np.array(res["err_shares"]) * 100.0
    totals = res["totals"]

    plt.figure(figsize=(12, 6))
    for c in range(err_counts.shape[1]):
        label_total = f" (N={totals[c]})" if totals[c] is not None else ""
        plt.plot(
            x,
            err_counts[:, c],
            marker="o",
            label=f"class {c}{label_total}",
        )
    plt.xlabel("config (fc1|fc2|fc3)")
    plt.ylabel("# Errors (absolute)")
    plt.title("Errors per Class vs Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "errors_per_class_counts.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    for c in range(err_shares.shape[1]):
        label_total = f" (N={totals[c]})" if totals[c] is not None else ""
        plt.plot(
            x,
            err_shares[:, c],
            marker="o",
            label=f"class {c}{label_total}",
        )
    plt.xlabel("config (fc1|fc2|fc3)")
    plt.ylabel("Error Share (%)")
    plt.title("Error Share per Class vs Quantization Config")
    plt.xticks(x, labels, rotation=90)
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "errors_per_class_pct.png"))
    plt.close()

    sort_idx = np.argsort(compr)
    compr_sorted = compr[sort_idx]
    ixz_sorted = ixz[sort_idx]
    izy_sorted = izy[sort_idx]
    accs_sorted = accs[sort_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(
        compr_sorted,
        ixz_sorted,
        marker="o",
    )
    plt.xlabel("Compression Ratio (compressed/original)")
    plt.ylabel("I(X;Z) (bits)")
    plt.title("I(X;Z) vs Compression Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ixz_vs_compression_ratio.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        compr_sorted,
        izy_sorted,
        marker="o",
    )
    plt.xlabel("Compression Ratio (compressed/original)")
    plt.ylabel("I(Z;Y) (bits)")
    plt.title("I(Z;Y) vs Compression Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "izy_vs_compression_ratio.png"))
    plt.close()

    theo_acc_ratio = fano_upper_accuracy_from_I(
        izy,
        K=10,
        H_Y_bits=math.log2(10),
    )
    theo_acc_ratio = 100.0 * np.array(theo_acc_ratio)
    theo_acc_ratio_sorted = theo_acc_ratio[sort_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(
        compr_sorted,
        100.0 * accs_sorted,
        marker="o",
        label="Actual Accuracy (%)",
    )
    plt.plot(
        compr_sorted,
        theo_acc_ratio_sorted,
        marker="s",
        label="Fano Upper Bound (%)",
    )
    plt.xlabel("Compression Ratio (compressed/original)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Compression Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_compression_ratio.png"))
    plt.close()


if __name__ == "__main__":
    main()
