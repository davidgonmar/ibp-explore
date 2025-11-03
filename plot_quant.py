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

    accs_by_decoder = res["accs"]
    izy_arr = np.array(res["izy"])

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
        labels_list.append(f"{dec_name} Accuracy (%)")

    (h_theo,) = ax1.plot(
        izy_sorted,
        theo_sorted,
        marker="s",
        label="Fano Upper Bound (%)",
    )
    handles.append(h_theo)
    labels_list.append("Fano Upper Bound (%)")

    ax1.set_xlabel("I(Z;Y) (bits)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy vs I(Z;Y)")
    ax1.grid(True)
    ax1.legend(handles, labels_list, loc="best")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "accuracy_vs_izy.pdf"),
        format="pdf",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
