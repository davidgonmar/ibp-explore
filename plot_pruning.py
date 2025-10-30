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

    results_dict = load_results_json(args.input_json)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for strategy_name, res in results_dict.items():
        plt.plot(res["ratios"], res["accs"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_ratio_all_methods.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    for strategy_name, res in results_dict.items():
        plt.plot(res["ratios"], res["ixz"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("I(X;Z) (bits)")
    plt.title("I(X;Z) vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ixz_vs_ratio_all_methods.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    for strategy_name, res in results_dict.items():
        plt.plot(res["ratios"], res["izy"], marker="o", label=strategy_name)
    plt.xlabel("Prune Ratio")
    plt.ylabel("I(Z;Y) (bits)")
    plt.title("I(Z;Y) vs Prune Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "izy_vs_ratio_all_methods.png"))
    plt.close()

    for strategy_name, res in results_dict.items():
        compr_arr = np.array(res["compr"])
        ixz_arr = np.array(res["ixz"])
        sort_idx = np.argsort(compr_arr)
        compr_sorted = compr_arr[sort_idx]
        ixz_sorted = ixz_arr[sort_idx]
        plt.figure(figsize=(8, 6))
        plt.plot(compr_sorted, ixz_sorted, marker="o")
        plt.xlabel("Compression Ratio (compressed/original)")
        plt.ylabel("I(X;Z) (bits)")
        plt.title(f"I(X;Z) vs Compression Ratio - {strategy_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"ixz_vs_compression_ratio_{strategy_name}.png",
            )
        )
        plt.close()

    for strategy_name, res in results_dict.items():
        compr_arr = np.array(res["compr"])
        izy_arr = np.array(res["izy"])
        sort_idx = np.argsort(compr_arr)
        compr_sorted = compr_arr[sort_idx]
        izy_sorted = izy_arr[sort_idx]
        plt.figure(figsize=(8, 6))
        plt.plot(compr_sorted, izy_sorted, marker="o")
        plt.xlabel("Compression Ratio (compressed/original)")
        plt.ylabel("I(Z;Y) (bits)")
        plt.title(f"I(Z;Y) vs Compression Ratio - {strategy_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"izy_vs_compression_ratio_{strategy_name}.png",
            )
        )
        plt.close()

    for strategy_name, res in results_dict.items():
        counts_arr = np.array(res["err_counts"])
        totals = (
            res["totals"]
            if res.get("totals") is not None
            else [None] * counts_arr.shape[1]
        )
        plt.figure(figsize=(9, 6))
        for c in range(counts_arr.shape[1]):
            label_total = f" (N={totals[c]})" if totals[c] is not None else ""
            plt.plot(
                res["ratios"],
                counts_arr[:, c],
                marker="o",
                label=f"class {c}{label_total}",
            )
        plt.xlabel("Prune Ratio")
        plt.ylabel("# Errors (absolute)")
        plt.title(f"Errors per Class vs Prune Ratio - {strategy_name}")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"errors_per_class_counts_{strategy_name}.png",
            )
        )
        plt.close()

    for strategy_name, res in results_dict.items():
        shares_arr = np.array(res["err_shares"]) * 100.0
        totals = (
            res["totals"]
            if res.get("totals") is not None
            else [None] * shares_arr.shape[1]
        )
        plt.figure(figsize=(9, 6))
        for c in range(shares_arr.shape[1]):
            label_total = f" (N={totals[c]})" if totals[c] is not None else ""
            plt.plot(
                res["ratios"],
                shares_arr[:, c],
                marker="o",
                label=f"class {c}{label_total}",
            )
        plt.xlabel("Prune Ratio")
        plt.ylabel("Error Share (%)")
        plt.title(f"Error Share per Class vs Prune Ratio - {strategy_name}")
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"errors_per_class_percentage_{strategy_name}.png",
            )
        )
        plt.close()

    for strategy_name, res in results_dict.items():
        ratios = res["ratios"]
        acc_pct = 100.0 * np.array(res["accs"])
        I_vals = np.array(res["izy"])

        theo_acc = fano_upper_accuracy_from_I(
            I_vals,
            K=10,
            H_Y_bits=math.log2(10),
        )
        theo_pct = 100.0 * np.array(theo_acc)

        plt.figure(figsize=(8.5, 6))
        plt.plot(
            ratios,
            acc_pct,
            marker="o",
            label="Actual Accuracy (%)",
        )
        plt.plot(
            ratios,
            theo_pct,
            marker="s",
            label="Theoretical (Fano upper bound) (%)",
        )
        plt.xlabel("Prune Ratio")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Actual vs Theoretical Accuracy - {strategy_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"accuracy_vs_theoretical_{strategy_name}.png",
            )
        )
        plt.close()

    for strategy_name, res in results_dict.items():
        compr_arr = np.array(res["compr"])
        acc_pct = 100.0 * np.array(res["accs"])
        I_vals = np.array(res["izy"])
        theo_acc = fano_upper_accuracy_from_I(
            I_vals,
            K=10,
            H_Y_bits=math.log2(10),
        )
        theo_pct = 100.0 * np.array(theo_acc)

        sort_idx = np.argsort(compr_arr)
        compr_sorted = compr_arr[sort_idx]
        acc_sorted = acc_pct[sort_idx]
        theo_sorted = theo_pct[sort_idx]

        plt.figure(figsize=(8.5, 6))
        plt.plot(
            compr_sorted,
            acc_sorted,
            marker="o",
            label="Actual Accuracy (%)",
        )
        plt.plot(
            compr_sorted,
            theo_sorted,
            marker="s",
            label="Fano Upper Bound (%)",
        )
        plt.xlabel("Compression Ratio (compressed/original)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy vs Compression Ratio - {strategy_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"accuracy_vs_compression_ratio_{strategy_name}.png",
            )
        )
        plt.close()


if __name__ == "__main__":
    main()
