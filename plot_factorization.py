import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

        # --- Collage of FANO/per-class-accuracy plots ---
        # Plot per-class Fano upper bound (k=2) vs per_class_differentiate accuracy for ALL classes in a single collage PDF

        per_class_binary_info_list = mres.get("per_class_binary_info", [])
        accs_per_class_differentiate_dict = mres.get("accs_per_class_differentiate", {})
        compr_list = mres.get("compr", [])

        # Handle both old format (list) and new format (dict by decoder)
        if isinstance(accs_per_class_differentiate_dict, list):
            accs_per_class_differentiate_dict = {
                "original": accs_per_class_differentiate_dict
            }

        if (
            per_class_binary_info_list
            and accs_per_class_differentiate_dict
            and compr_list
        ):
            first_info = per_class_binary_info_list[0]
            class_indices = sorted([int(k) for k in first_info.keys()])
            n_classes = len(class_indices)
            ncols = 5 if n_classes >= 5 else n_classes
            nrows = int(np.ceil(n_classes / ncols))

            compr_arr = np.array(compr_list[1:])

            # Prepare the collage PDF file
            collage_pdf_path = os.path.join(
                args.out_dir, f"per_class_fano_upper_{method_name}_collage.pdf"
            )
            pdf_pages = PdfPages(collage_pdf_path)

            # Optionally aggregate all subplots into a single grid page first
            fig_all, axes_all = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), squeeze=False
            )
            colors = {"original": "blue", "linear": "green", "mlp": "orange"}
            markers = {"original": "o", "linear": "v", "mlp": "D"}

            for i, class_idx in enumerate(class_indices):
                # Fano & per-class differentiate accuracy
                i_x_w_k_list = []
                for per_class_binary_info in per_class_binary_info_list:
                    if str(class_idx) in per_class_binary_info:
                        i_x_w_k_list.append(
                            per_class_binary_info[str(class_idx)]["I_x_w_k"]
                        )
                    elif class_idx in per_class_binary_info:
                        i_x_w_k_list.append(per_class_binary_info[class_idx]["I_x_w_k"])
                    else:
                        i_x_w_k_list.append(0.0)  # fallback

                i_x_w_k_arr = np.array(i_x_w_k_list[1:])
                compr_arr_nonorig = compr_arr
                balance = True
                ent = (
                    -(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9))
                    if not balance
                    else 1.0
                )
                fano_acc = fano_upper_accuracy_from_I(i_x_w_k_arr, K=2, H_Y_bits=ent)
                fano_acc_pct = 100.0 * np.array(fano_acc)

                sort_idx = np.argsort(compr_arr_nonorig)
                compr_sorted = compr_arr_nonorig[sort_idx]
                fano_acc_sorted = fano_acc_pct[sort_idx]

                per_class_diff_by_decoder = {}
                for (
                    decoder_name,
                    accs_per_class_diff_list,
                ) in accs_per_class_differentiate_dict.items():
                    per_class_diff_list = []
                    for accs_per_class_diff in accs_per_class_diff_list:
                        if (
                            isinstance(accs_per_class_diff, list)
                            and len(accs_per_class_diff) > class_idx
                        ):
                            error_rate = accs_per_class_diff[class_idx]
                            accuracy = 1.0 - error_rate
                            per_class_diff_list.append(accuracy)
                        else:
                            per_class_diff_list.append(0.0)
                    per_class_diff_arr = np.array(per_class_diff_list[1:])
                    per_class_diff_pct = 100.0 * per_class_diff_arr
                    per_class_diff_by_decoder[decoder_name] = per_class_diff_pct[
                        sort_idx
                    ]

                # ============= Each subplot =============
                row = i // ncols
                col = i % ncols
                ax = axes_all[row][col]
                ax.plot(
                    compr_sorted,
                    fano_acc_sorted,
                    marker="s",
                    label="Fano UB (k=2) (%)",
                    color="red",
                    linewidth=2,
                )

                for (
                    decoder_name,
                    per_class_diff_sorted,
                ) in per_class_diff_by_decoder.items():
                    color = colors.get(decoder_name, "gray")
                    marker = markers.get(decoder_name, "o")
                    ax.plot(
                        compr_sorted,
                        per_class_diff_sorted,
                        marker=marker,
                        label=(
                            f"PC Diff Acc ({decoder_name}) (%)"
                            if row == 0 and col == 0
                            else None
                        ),  # avoid repeated labels
                        color=color,
                        linewidth=2,
                    )

                ax.set_xlabel("Compr. Ratio")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"Class {class_idx}", fontsize=11)
                ax.grid(True, alpha=0.3)
                if row == 0 and col == 0:  # legend only in first subplot
                    ax.legend(fontsize=8, loc="best")

            # Remove unused axes (if any)
            for j in range(n_classes, nrows * ncols):
                fig_all.delaxes(axes_all[j // ncols][j % ncols])

            plt.suptitle(
                f"{method_name}: Fano UB vs Per-Class Differentiate Acc. by Compression",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf_pages.savefig(fig_all)
            plt.close(fig_all)

            # Optionally, also save each as its own page for nicer big per-class charts:
            for i, class_idx in enumerate(class_indices):
                i_x_w_k_list = []
                for per_class_binary_info in per_class_binary_info_list:
                    if str(class_idx) in per_class_binary_info:
                        i_x_w_k_list.append(
                            per_class_binary_info[str(class_idx)]["I_x_w_k"]
                        )
                    elif class_idx in per_class_binary_info:
                        i_x_w_k_list.append(per_class_binary_info[class_idx]["I_x_w_k"])
                    else:
                        i_x_w_k_list.append(0.0)
                i_x_w_k_arr = np.array(i_x_w_k_list[1:])
                balance = True
                ent = (
                    -(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9))
                    if not balance
                    else 1.0
                )
                fano_acc = fano_upper_accuracy_from_I(i_x_w_k_arr, K=2, H_Y_bits=ent)
                fano_acc_pct = 100.0 * np.array(fano_acc)
                sort_idx = np.argsort(compr_arr)
                compr_sorted = compr_arr[sort_idx]
                fano_acc_sorted = fano_acc_pct[sort_idx]
                per_class_diff_by_decoder = {}
                for (
                    decoder_name,
                    accs_per_class_diff_list,
                ) in accs_per_class_differentiate_dict.items():
                    per_class_diff_list = []
                    for accs_per_class_diff in accs_per_class_diff_list:
                        if (
                            isinstance(accs_per_class_diff, list)
                            and len(accs_per_class_diff) > class_idx
                        ):
                            error_rate = accs_per_class_diff[class_idx]
                            accuracy = 1.0 - error_rate
                            per_class_diff_list.append(accuracy)
                        else:
                            per_class_diff_list.append(0.0)
                    per_class_diff_arr = np.array(per_class_diff_list[1:])
                    per_class_diff_pct = 100.0 * per_class_diff_arr
                    per_class_diff_by_decoder[decoder_name] = per_class_diff_pct[
                        sort_idx
                    ]
                fig, ax1 = plt.subplots(figsize=(7, 5))
                (h_fano,) = ax1.plot(
                    compr_sorted,
                    fano_acc_sorted,
                    marker="s",
                    label="Fano UB (k=2) (%)",
                    color="red",
                    linewidth=2,
                )
                for (
                    decoder_name,
                    per_class_diff_sorted,
                ) in per_class_diff_by_decoder.items():
                    color = colors.get(decoder_name, "gray")
                    marker = markers.get(decoder_name, "o")
                    ax1.plot(
                        compr_sorted,
                        per_class_diff_sorted,
                        marker=marker,
                        label=f"PC Diff Acc ({decoder_name}) (%)",
                        color=color,
                        linewidth=2,
                    )
                ax1.set_xlabel("Compression Ratio")
                ax1.set_ylabel("Accuracy (%)")
                ax1.set_title(
                    f"{method_name}: Class {class_idx} - Fano Bound & Per-Class Acc vs Compression",
                    fontsize=12,
                )
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc="best", fontsize=10)
                fig.tight_layout()
                pdf_pages.savefig(fig)
                plt.close(fig)
            pdf_pages.close()
            print(f"Saved collective collage PDF: {collage_pdf_path}")

            # Additionally, save individual PNGs/PDFs as usual (optional: comment if not needed)
            for i, class_idx in enumerate(class_indices):
                # Fano + per-class acc (single class)
                i_x_w_k_list = []
                for per_class_binary_info in per_class_binary_info_list:
                    if str(class_idx) in per_class_binary_info:
                        i_x_w_k_list.append(
                            per_class_binary_info[str(class_idx)]["I_x_w_k"]
                        )
                    elif class_idx in per_class_binary_info:
                        i_x_w_k_list.append(per_class_binary_info[class_idx]["I_x_w_k"])
                    else:
                        i_x_w_k_list.append(0.0)
                i_x_w_k_arr = np.array(i_x_w_k_list[1:])
                balance = True
                ent = (
                    -(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9))
                    if not balance
                    else 1.0
                )
                fano_acc = fano_upper_accuracy_from_I(i_x_w_k_arr, K=2, H_Y_bits=ent)
                fano_acc_pct = 100.0 * np.array(fano_acc)
                sort_idx = np.argsort(compr_arr)
                compr_sorted = compr_arr[sort_idx]
                fano_acc_sorted = fano_acc_pct[sort_idx]
                per_class_diff_by_decoder = {}
                for (
                    decoder_name,
                    accs_per_class_diff_list,
                ) in accs_per_class_differentiate_dict.items():
                    per_class_diff_list = []
                    for accs_per_class_diff in accs_per_class_diff_list:
                        if (
                            isinstance(accs_per_class_diff, list)
                            and len(accs_per_class_diff) > class_idx
                        ):
                            error_rate = accs_per_class_diff[class_idx]
                            accuracy = 1.0 - error_rate
                            per_class_diff_list.append(accuracy)
                        else:
                            per_class_diff_list.append(0.0)
                    per_class_diff_arr = np.array(per_class_diff_list[1:])
                    per_class_diff_pct = 100.0 * per_class_diff_arr
                    per_class_diff_by_decoder[decoder_name] = per_class_diff_pct[
                        sort_idx
                    ]
                # Plot Fano upper + per-class accuracy (one big PDF per class)
                fig_indiv, ax_indiv = plt.subplots(figsize=(12, 8))
                (h_fano,) = ax_indiv.plot(
                    compr_sorted,
                    fano_acc_sorted,
                    marker="s",
                    label="Fano UB (k=2) (%)",
                    color="red",
                    linewidth=2,
                )
                for (
                    decoder_name,
                    per_class_diff_sorted,
                ) in per_class_diff_by_decoder.items():
                    color = colors.get(decoder_name, "gray")
                    marker = markers.get(decoder_name, "o")
                    ax_indiv.plot(
                        compr_sorted,
                        per_class_diff_sorted,
                        marker=marker,
                        label=f"Per-Class Differentiate Accuracy ({decoder_name}) (%)",
                        color=color,
                        linewidth=2,
                    )
                ax_indiv.set_xlabel(
                    "Compression Ratio (params after / params before)", fontsize=12
                )
                ax_indiv.set_ylabel("Accuracy (%)", fontsize=12)
                ax_indiv.set_title(
                    f"{method_name}: Class {class_idx} - Fano Bound vs Per-Class Accuracy vs Compression Ratio",
                    fontsize=14,
                )
                ax_indiv.grid(True, alpha=0.3)
                ax_indiv.legend(loc="best", fontsize=10)
                fig_indiv.tight_layout()
                fig_indiv.savefig(
                    os.path.join(
                        args.out_dir,
                        f"per_class_fano_upper_{method_name}_class_{class_idx}_vs_compr.pdf",
                    ),
                    format="pdf",
                )
                plt.close(fig_indiv)
                # ALSO: Plot per-class info (I_x_w_k) vs per-class differentiate accuracy, x-axis: compression ratio
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                ax2.plot(
                    compr_sorted,
                    i_x_w_k_arr[sort_idx],
                    marker="^",
                    label="I(X;w_k) (bits)",
                    color="green",
                    linewidth=2,
                )
                ax2.set_ylabel("I(X;w_k) (bits)", color="green", fontsize=12)
                ax3 = ax2.twinx()
                for (
                    decoder_name,
                    per_class_diff_sorted,
                ) in per_class_diff_by_decoder.items():
                    color = colors.get(decoder_name, "gray")
                    marker = markers.get(decoder_name, "o")
                    ax3.plot(
                        compr_sorted,
                        per_class_diff_sorted,
                        marker=marker,
                        label=f"Per-Class Differentiate Accuracy ({decoder_name}) (%)",
                        color=color,
                        linewidth=2,
                    )
                ax3.set_ylabel("Accuracy (%)", color="blue", fontsize=12)
                ax2.set_xlabel(
                    "Compression Ratio (params after / params before)", fontsize=12
                )
                ax2.set_title(
                    f"{method_name}: Class {class_idx} - I(X;w_k) and Per-Class Accuracy vs Compression Ratio",
                    fontsize=14,
                )
                ax2.grid(True, alpha=0.3)
                lines_1, labels_1 = ax2.get_legend_handles_labels()
                lines_2, labels_2 = ax3.get_legend_handles_labels()
                ax2.legend(
                    lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=10
                )
                fig2.tight_layout()
                fig2.savefig(
                    os.path.join(
                        args.out_dir,
                        f"per_class_info_acc_{method_name}_class_{class_idx}_vs_compr.pdf",
                    ),
                    format="pdf",
                )
                plt.close(fig2)
        # ============== End collage code ===============


if __name__ == "__main__":
    main()
