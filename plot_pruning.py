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

    for strategy_name, res in results_dict.items():
        izy_arr = np.array(res["izy"])
        accs_by_decoder = res["accs"]

        theo_acc = fano_upper_accuracy_from_I(
            izy_arr,
            K=10,
            H_Y_bits=math.log2(10),
        )
        theo_pct = 100.0 * np.array(theo_acc)

        sort_idx = np.argsort(izy_arr)
        izy_sorted = izy_arr[sort_idx]
        theo_sorted = theo_pct[sort_idx]

        fig, ax1 = plt.subplots(figsize=(8.5, 6))

        handles = []
        labels = []

        for dec_name, acc_list in accs_by_decoder.items():
            acc_pct = 100.0 * np.array(acc_list)
            acc_sorted = acc_pct[sort_idx]
            (h,) = ax1.plot(
                izy_sorted,
                acc_sorted,
                marker="o",
                label=f"{dec_name} Accuracy (%)",
            )
            handles.append(h)
            labels.append(f"{dec_name} Accuracy (%)")

        (h_theo,) = ax1.plot(
            izy_sorted,
            theo_sorted,
            marker="s",
            label="Fano Upper Bound (%)",
        )
        handles.append(h_theo)
        labels.append("Fano Upper Bound (%)")

        ax1.set_xlabel("I(Z;Y) (bits)")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title(f"Accuracy vs I(Z;Y) - {strategy_name}")
        ax1.grid(True)
        ax1.legend(handles, labels)
        fig.tight_layout()

        fig.savefig(
            os.path.join(
                out_dir,
                f"accuracy_vs_izy_{strategy_name}.pdf",
            ),
            format="pdf",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
