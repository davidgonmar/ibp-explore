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
        izy_full = np.array(mres["izy"])
        accs_full = mres["accs"]

        izy_arr = izy_full[1:]

        theo_acc = fano_upper_accuracy_from_I(
            izy_arr,
            K=10,
            H_Y_bits=math.log2(10),
        )
        theo_pct = 100.0 * np.array(theo_acc)

        sort_idx = np.argsort(izy_arr)
        izy_sorted = izy_arr[sort_idx]
        theo_sorted = theo_pct[sort_idx]

        fig, ax1 = plt.subplots(figsize=(12, 8))

        handles = []
        labels_list = []

        for probe_name, probe_acc_list in accs_full.items():
            probe_acc_arr = np.array(probe_acc_list)[1:]
            probe_acc_pct = 100.0 * probe_acc_arr
            probe_acc_sorted = probe_acc_pct[sort_idx]
            (h_probe,) = ax1.plot(
                izy_sorted,
                probe_acc_sorted,
                marker="o",
                label=f"{probe_name} Acc (%)",
            )
            handles.append(h_probe)
            labels_list.append(f"{probe_name} Acc (%)")

        (h_theo,) = ax1.plot(
            izy_sorted,
            theo_sorted,
            marker="s",
            label="Fano (%)",
        )
        handles.append(h_theo)
        labels_list.append("Fano (%)")

        ax1.set_xlabel("I(Z;Y) (bits)")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title(f"{method_name}: Accuracy vs I(Z;Y)")
        ax1.grid(True)
        ax1.legend(handles, labels_list, loc="best")

        fig.tight_layout()
        fig.savefig(
            os.path.join(args.out_dir, f"accuracy_vs_izy_{method_name}.pdf"),
            format="pdf",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
