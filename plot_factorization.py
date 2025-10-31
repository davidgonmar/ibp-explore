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
    os.makedirs(args.out_dir, exist_ok=True)

    methods = res["methods"]

    for method_name, mres in methods.items():
        labels_full = mres["labels"]
        ixz_full = np.array(mres["ixz"])
        izy_full = np.array(mres["izy"])
        compr_full = np.array(mres["compr"])
        accs_full = mres["accs"]

        labels_plot = labels_full[1:]
        ixz_arr = ixz_full[1:]
        izy_arr = izy_full[1:]
        compr_arr = compr_full[1:]

        theo_acc = fano_upper_accuracy_from_I(
            izy_arr,
            K=10,
            H_Y_bits=math.log2(10),
        )
        theo_pct = 100.0 * np.array(theo_acc)

        x = np.arange(len(labels_plot))

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        handles = []
        labels_list = []

        for probe_name, probe_acc_list in accs_full.items():
            probe_acc_arr = np.array(probe_acc_list)[1:]
            probe_acc_pct = 100.0 * probe_acc_arr
            (h_probe,) = ax1.plot(
                x,
                probe_acc_pct,
                marker="o",
                label=f"{probe_name} Acc (%)",
            )
            handles.append(h_probe)
            labels_list.append(f"{probe_name} Acc (%)")

        (h_theo,) = ax1.plot(
            x,
            theo_pct,
            marker="s",
            label="Fano (%)",
        )
        handles.append(h_theo)
        labels_list.append("Fano (%)")

        (h_ixz,) = ax2.plot(
            x,
            ixz_arr,
            marker="^",
            linestyle="--",
            label="I(X;Z)",
        )
        handles.append(h_ixz)
        labels_list.append("I(X;Z)")

        ax1.set_xlabel("Factorization Target (rank ratio kept)")
        ax1.set_ylabel("Accuracy (%)")
        ax2.set_ylabel("I(X;Z) (bits)")
        ax1.set_title(
            f"{method_name}: Accuracy, Fano bound, and I(X;Z) vs Rank Ratio Kept"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels_plot, rotation=90)
        ax1.grid(True)
        ax1.legend(handles, labels_list, loc="best")

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                args.out_dir, f"accuracy_ixz_vs_factorization_{method_name}.pdf"
            ),
            format="pdf",
        )
        plt.close(fig)

        sort_idx = np.argsort(compr_arr)
        compr_sorted = compr_arr[sort_idx]
        ixz_sorted = ixz_arr[sort_idx]
        theo_sorted = theo_pct[sort_idx]

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()

        handles = []
        labels_list = []

        for probe_name, probe_acc_list in accs_full.items():
            probe_acc_arr = np.array(probe_acc_list)[1:]
            probe_acc_pct = 100.0 * probe_acc_arr
            probe_acc_sorted = probe_acc_pct[sort_idx]
            (h_probe,) = ax1.plot(
                compr_sorted,
                probe_acc_sorted,
                marker="o",
                label=f"{probe_name} Acc (%)",
            )
            handles.append(h_probe)
            labels_list.append(f"{probe_name} Acc (%)")

        theo_sorted_acc = theo_sorted
        (h_theo,) = ax1.plot(
            compr_sorted,
            theo_sorted_acc,
            marker="s",
            label="Fano (%)",
        )
        handles.append(h_theo)
        labels_list.append("Fano (%)")

        (h_ixz,) = ax2.plot(
            compr_sorted,
            ixz_sorted,
            marker="^",
            linestyle="--",
            label="I(X;Z)",
        )
        handles.append(h_ixz)
        labels_list.append("I(X;Z)")

        ax1.set_xlabel("Compression Ratio (compressed/original)")
        ax1.set_ylabel("Accuracy (%)")
        ax2.set_ylabel("I(X;Z) (bits)")
        ax1.set_title(
            f"{method_name}: Accuracy, Fano bound, and I(X;Z) vs Compression Ratio"
        )
        ax1.grid(True)
        ax1.legend(handles, labels_list, loc="best")

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                args.out_dir, f"accuracy_ixz_vs_compression_ratio_{method_name}.pdf"
            ),
            format="pdf",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
