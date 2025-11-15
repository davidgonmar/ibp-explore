import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import load_results_json, fano_upper_accuracy_from_I


def get_distinct_colors(n_lines):
    cmap_names = ["tab20", "tab20b", "tab20c"]
    all_colors = []
    for cmap_name in cmap_names:
        cmap = plt.get_cmap(cmap_name)
        all_colors.extend([cmap(i) for i in range(cmap.N)])
    # flatten and ensure uniqueness but preserve order
    seen = set()
    unique_colors = []
    for color in all_colors:
        if color not in seen:
            unique_colors.append(color)
            seen.add(color)
    if n_lines > len(unique_colors):
        prop_cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        while len(unique_colors) < n_lines:
            for c in prop_cycle_colors:
                col = plt.colors.to_rgba(c)
                if col not in seen:
                    unique_colors.append(col)
                    seen.add(col)
                if len(unique_colors) >= n_lines:
                    break
    return unique_colors[:n_lines]


def get_distinct_styles(n_lines, kind="main"):
    colors = get_distinct_colors(n_lines)
    linestyles = ["-"] * n_lines
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "*",
        "P",
        "X",
        "<",
        ">",
        "H",
        "+",
        "x",
        "1",
        "2",
    ]
    styles = []
    for i in range(n_lines):
        styles.append(
            dict(
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
            )
        )
    return styles


def plot_data_with_before_after(
    ax,
    x_before,
    y_before,
    x_after,
    y_after,
    label_base,
    marker_before="o",
    marker_after="s",
    color_before="blue",
    color_after="red",
    linestyle_before="-",
    linestyle_after="--",
):
    # This function is not used below, but keep as is (for completeness)
    h_before = ax.plot(
        x_before,
        y_before,
        marker=marker_before,
        label=f"{label_base} (Before Retrain)",
        color=color_before,
        linestyle=linestyle_before,
        linewidth=2,
    )
    h_after = ax.plot(
        x_after,
        y_after,
        marker=marker_after,
        label=f"{label_base} (After Retrain)",
        color=color_after,
        linestyle=linestyle_after,
        linewidth=2,
    )
    return h_before[0], h_after[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    res = load_results_json(args.input_json)
    os.makedirs(args.out_dir, exist_ok=True)

    methods = res["methods"]

    for method_name, mres in methods.items():
        has_retrain_data = "before_retrain" in mres and "after_retrain" in mres
        if not has_retrain_data:
            print(
                f"Warning: {method_name} does not have before_retrain/after_retrain data. Skipping."
            )
            continue

        before_data = mres["before_retrain"]
        after_data = mres["after_retrain"]

        # --- Shared prep ---
        izy_before_full = np.array(before_data["izy"])
        izy_after_full = np.array(after_data["izy"])
        izy_before_arr = izy_before_full[1:]
        izy_after_arr = izy_after_full[1:]

        theo_acc_before = fano_upper_accuracy_from_I(
            izy_before_arr, K=10, H_Y_bits=math.log2(10)
        )
        theo_pct_before = 100.0 * np.array(theo_acc_before)
        theo_acc_after = fano_upper_accuracy_from_I(
            izy_after_arr, K=10, H_Y_bits=math.log2(10)
        )
        theo_pct_after = 100.0 * np.array(theo_acc_after)

        sort_idx_before = np.argsort(izy_before_arr)
        izy_sorted_before = izy_before_arr[sort_idx_before]
        theo_sorted_before = theo_pct_before[sort_idx_before]
        sort_idx_after = np.argsort(izy_after_arr)
        izy_sorted_after = izy_after_arr[sort_idx_after]
        theo_sorted_after = theo_pct_after[sort_idx_after]

        accs_before = before_data["accs"]
        accs_after = after_data["accs"]
        probe_names = list(accs_before.keys())
        n_lines_acc = len(probe_names) * 2 + 2
        probe_styles = get_distinct_styles(n_lines_acc)

        # --- Plot Both Before/After In One Plot (as before) ---
        fig, ax1 = plt.subplots(figsize=(12, 8))
        handles = []
        labels_list = []

        for i, probe_name in enumerate(probe_names):
            style_before = probe_styles[2 * i]
            style_after = probe_styles[2 * i + 1]
            probe_acc_before_arr = np.array(accs_before[probe_name])[1:]
            probe_acc_after_arr = np.array(accs_after[probe_name])[1:]
            probe_acc_before_pct = 100.0 * probe_acc_before_arr
            probe_acc_after_pct = 100.0 * probe_acc_after_arr
            probe_acc_sorted_before = probe_acc_before_pct[sort_idx_before]
            probe_acc_sorted_after = probe_acc_after_pct[sort_idx_after]
            (h_before,) = ax1.plot(
                izy_sorted_before,
                probe_acc_sorted_before,
                label=f"{probe_name} Acc (%) (Before)",
                color=style_before["color"],
                linestyle=style_before["linestyle"],
                marker=style_before["marker"],
                linewidth=2,
            )
            (h_after,) = ax1.plot(
                izy_sorted_after,
                probe_acc_sorted_after,
                label=f"{probe_name} Acc (%) (After)",
                color=style_after["color"],
                linestyle=style_after["linestyle"],
                marker=style_after["marker"],
                linewidth=2,
            )
            handles.extend([h_before, h_after])
            labels_list.extend(
                [f"{probe_name} Acc (%) (Before)", f"{probe_name} Acc (%) (After)"]
            )

        # Fano bounds
        style_fano_before = probe_styles[len(probe_names) * 2]
        style_fano_after = probe_styles[len(probe_names) * 2 + 1]
        (h_fano_before,) = ax1.plot(
            izy_sorted_before,
            theo_sorted_before,
            label="Fano (%) (Before Retrain)",
            color=style_fano_before["color"],
            linestyle=style_fano_before["linestyle"],
            marker=style_fano_before["marker"],
            linewidth=2,
        )
        (h_fano_after,) = ax1.plot(
            izy_sorted_after,
            theo_sorted_after,
            label="Fano (%) (After Retrain)",
            color=style_fano_after["color"],
            linestyle=style_fano_after["linestyle"],
            marker=style_fano_after["marker"],
            linewidth=2,
        )
        handles.extend([h_fano_before, h_fano_after])
        labels_list.extend(["Fano (%) (Before Retrain)", "Fano (%) (After Retrain)"])

        ax1.set_xlabel("I(Z;Y) (bits)")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title(f"{method_name}: Accuracy vs I(Z;Y) (Before/After Retrain)")
        ax1.grid(True)
        ax1.legend(handles, labels_list, loc="best")
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.out_dir, f"accuracy_vs_izy_{method_name}.pdf"),
            format="pdf",
        )
        plt.close(fig)

        # --- Separate "Before Retrain" Plot ---
        fig_before, ax1_before = plt.subplots(figsize=(12, 8))
        handles_before = []
        labels_before = []
        for i, probe_name in enumerate(probe_names):
            style_before = probe_styles[2 * i]
            probe_acc_before_arr = np.array(accs_before[probe_name])[1:]
            probe_acc_before_pct = 100.0 * probe_acc_before_arr
            probe_acc_sorted_before = probe_acc_before_pct[sort_idx_before]
            (h_probe,) = ax1_before.plot(
                izy_sorted_before,
                probe_acc_sorted_before,
                label=f"{probe_name} Acc (%)",
                color=style_before["color"],
                linestyle=style_before["linestyle"],
                marker=style_before["marker"],
                linewidth=2,
            )
            handles_before.append(h_probe)
            labels_before.append(f"{probe_name} Acc (%)")

        (h_fano,) = ax1_before.plot(
            izy_sorted_before,
            theo_sorted_before,
            label="Fano Bound (%)",
            color=style_fano_before["color"],
            linestyle=style_fano_before["linestyle"],
            marker=style_fano_before["marker"],
            linewidth=2,
        )
        handles_before.append(h_fano)
        labels_before.append("Fano Bound (%)")
        ax1_before.set_xlabel("I(Z;Y) (bits)")
        ax1_before.set_ylabel("Accuracy (%)")
        ax1_before.set_title(f"{method_name}: Accuracy vs I(Z;Y) (Before Retrain)")
        ax1_before.grid(True)
        ax1_before.legend(handles_before, labels_before, loc="best")
        fig_before.tight_layout()
        fig_before.savefig(
            os.path.join(args.out_dir, f"accuracy_vs_izy_{method_name}_before.pdf"),
            format="pdf",
        )
        plt.close(fig_before)

        # --- Separate "After Retrain" Plot ---
        fig_after, ax1_after = plt.subplots(figsize=(12, 8))
        handles_after = []
        labels_after = []
        for i, probe_name in enumerate(probe_names):
            style_after = probe_styles[2 * i + 1]
            probe_acc_after_arr = np.array(accs_after[probe_name])[1:]
            probe_acc_after_pct = 100.0 * probe_acc_after_arr
            probe_acc_sorted_after = probe_acc_after_pct[sort_idx_after]
            (h_probe,) = ax1_after.plot(
                izy_sorted_after,
                probe_acc_sorted_after,
                label=f"{probe_name} Acc (%)",
                color=style_after["color"],
                linestyle=style_after["linestyle"],
                marker=style_after["marker"],
                linewidth=2,
            )
            handles_after.append(h_probe)
            labels_after.append(f"{probe_name} Acc (%)")
        (h_fano,) = ax1_after.plot(
            izy_sorted_after,
            theo_sorted_after,
            label="Fano Bound (%)",
            color=style_fano_after["color"],
            linestyle=style_fano_after["linestyle"],
            marker=style_fano_after["marker"],
            linewidth=2,
        )
        handles_after.append(h_fano)
        labels_after.append("Fano Bound (%)")
        ax1_after.set_xlabel("I(Z;Y) (bits)")
        ax1_after.set_ylabel("Accuracy (%)")
        ax1_after.set_title(f"{method_name}: Accuracy vs I(Z;Y) (After Retrain)")
        ax1_after.grid(True)
        ax1_after.legend(handles_after, labels_after, loc="best")
        fig_after.tight_layout()
        fig_after.savefig(
            os.path.join(args.out_dir, f"accuracy_vs_izy_{method_name}_after.pdf"),
            format="pdf",
        )
        plt.close(fig_after)

        # --- Accuracy vs Compression plot ---
        compr_list_before = before_data.get("compr", [])
        compr_list_after = after_data.get("compr", [])

        if compr_list_before and compr_list_after:
            compr_arr = np.array(compr_list_before[1:])

            theo_acc_compr_before = fano_upper_accuracy_from_I(
                izy_before_arr,
                K=10,
                H_Y_bits=math.log2(10),
            )
            theo_pct_compr_before = 100.0 * np.array(theo_acc_compr_before)

            theo_acc_compr_after = fano_upper_accuracy_from_I(
                izy_after_arr,
                K=10,
                H_Y_bits=math.log2(10),
            )
            theo_pct_compr_after = 100.0 * np.array(theo_acc_compr_after)

            sort_idx_compr = np.argsort(compr_arr)
            compr_sorted = compr_arr[sort_idx_compr]
            theo_sorted_compr_before = theo_pct_compr_before[sort_idx_compr]
            theo_sorted_compr_after = theo_pct_compr_after[sort_idx_compr]

            n_lines_compr = len(probe_names) * 2 + 2
            probe_styles_compr = get_distinct_styles(n_lines_compr)

            # Plot combined (before/after) accuracy vs compression, as before
            fig_compr, ax1_compr = plt.subplots(figsize=(12, 8))
            handles_compr = []
            labels_list_compr = []
            for i, probe_name in enumerate(probe_names):
                style_before = probe_styles_compr[2 * i]
                style_after = probe_styles_compr[2 * i + 1]

                probe_acc_before_arr = np.array(accs_before[probe_name])[1:]
                probe_acc_after_arr = np.array(accs_after[probe_name])[1:]
                probe_acc_before_pct = 100.0 * probe_acc_before_arr
                probe_acc_after_pct = 100.0 * probe_acc_after_arr
                probe_acc_sorted_before = probe_acc_before_pct[sort_idx_compr]
                probe_acc_sorted_after = probe_acc_after_pct[sort_idx_compr]

                (h_before,) = ax1_compr.plot(
                    compr_sorted,
                    probe_acc_sorted_before,
                    label=f"{probe_name} Accuracy (%) (Before)",
                    color=style_before["color"],
                    linestyle=style_before["linestyle"],
                    marker=style_before["marker"],
                    linewidth=2,
                )
                (h_after,) = ax1_compr.plot(
                    compr_sorted,
                    probe_acc_sorted_after,
                    label=f"{probe_name} Accuracy (%) (After)",
                    color=style_after["color"],
                    linestyle=style_after["linestyle"],
                    marker=style_after["marker"],
                    linewidth=2,
                )
                handles_compr.extend([h_before, h_after])
                labels_list_compr.extend(
                    [
                        f"{probe_name} Accuracy (%) (Before)",
                        f"{probe_name} Accuracy (%) (After)",
                    ]
                )

            style_fano_before = probe_styles_compr[len(probe_names) * 2]
            style_fano_after = probe_styles_compr[len(probe_names) * 2 + 1]
            (h_fano_before,) = ax1_compr.plot(
                compr_sorted,
                theo_sorted_compr_before,
                label="Fano Upper Bound (%) (Before Retrain)",
                color=style_fano_before["color"],
                linestyle=style_fano_before["linestyle"],
                marker=style_fano_before["marker"],
                linewidth=2,
            )
            (h_fano_after,) = ax1_compr.plot(
                compr_sorted,
                theo_sorted_compr_after,
                label="Fano Upper Bound (%) (After Retrain)",
                color=style_fano_after["color"],
                linestyle=style_fano_after["linestyle"],
                marker=style_fano_after["marker"],
                linewidth=2,
            )
            handles_compr.extend([h_fano_before, h_fano_after])
            labels_list_compr.extend(
                [
                    "Fano Upper Bound (%) (Before Retrain)",
                    "Fano Upper Bound (%) (After Retrain)",
                ]
            )

            ax1_compr.set_xlabel("Compression Ratio")
            ax1_compr.set_ylabel("Accuracy (%)")
            ax1_compr.set_title(
                f"{method_name}: Accuracy vs Compression (Before/After Retrain)"
            )
            ax1_compr.grid(True)
            ax1_compr.legend(handles_compr, labels_list_compr, loc="best")
            fig_compr.tight_layout()
            fig_compr.savefig(
                os.path.join(args.out_dir, f"accuracy_vs_compr_{method_name}.pdf"),
                format="pdf",
            )
            plt.close(fig_compr)

            # --- Separate "Before Retrain" Accuracy vs Compression Plot ---
            fig_compr_before, ax1_compr_before = plt.subplots(figsize=(12, 8))
            handles_cbefore = []
            labels_cbefore = []
            for i, probe_name in enumerate(probe_names):
                style_before = probe_styles_compr[2 * i]
                probe_acc_before_arr = np.array(accs_before[probe_name])[1:]
                probe_acc_before_pct = 100.0 * probe_acc_before_arr
                probe_acc_sorted_before = probe_acc_before_pct[sort_idx_compr]
                (h_probe,) = ax1_compr_before.plot(
                    compr_sorted,
                    probe_acc_sorted_before,
                    label=f"{probe_name} Accuracy (%)",
                    color=style_before["color"],
                    linestyle=style_before["linestyle"],
                    marker=style_before["marker"],
                    linewidth=2,
                )
                handles_cbefore.append(h_probe)
                labels_cbefore.append(f"{probe_name} Accuracy (%)")

            (h_fano,) = ax1_compr_before.plot(
                compr_sorted,
                theo_sorted_compr_before,
                label="Fano Bound (%)",
                color=style_fano_before["color"],
                linestyle=style_fano_before["linestyle"],
                marker=style_fano_before["marker"],
                linewidth=2,
            )
            handles_cbefore.append(h_fano)
            labels_cbefore.append("Fano Bound (%)")
            ax1_compr_before.set_xlabel("Compression Ratio")
            ax1_compr_before.set_ylabel("Accuracy (%)")
            ax1_compr_before.set_title(
                f"{method_name}: Accuracy vs Compression (Before Retrain)"
            )
            ax1_compr_before.grid(True)
            ax1_compr_before.legend(handles_cbefore, labels_cbefore, loc="best")
            fig_compr_before.tight_layout()
            fig_compr_before.savefig(
                os.path.join(
                    args.out_dir, f"accuracy_vs_compr_{method_name}_before.pdf"
                ),
                format="pdf",
            )
            plt.close(fig_compr_before)

            # --- Separate "After Retrain" Accuracy vs Compression Plot ---
            fig_compr_after, ax1_compr_after = plt.subplots(figsize=(12, 8))
            handles_cafter = []
            labels_cafter = []
            for i, probe_name in enumerate(probe_names):
                style_after = probe_styles_compr[2 * i + 1]
                probe_acc_after_arr = np.array(accs_after[probe_name])[1:]
                probe_acc_after_pct = 100.0 * probe_acc_after_arr
                probe_acc_sorted_after = probe_acc_after_pct[sort_idx_compr]
                (h_probe,) = ax1_compr_after.plot(
                    compr_sorted,
                    probe_acc_sorted_after,
                    label=f"{probe_name} Accuracy (%)",
                    color=style_after["color"],
                    linestyle=style_after["linestyle"],
                    marker=style_after["marker"],
                    linewidth=2,
                )
                handles_cafter.append(h_probe)
                labels_cafter.append(f"{probe_name} Accuracy (%)")
            (h_fano,) = ax1_compr_after.plot(
                compr_sorted,
                theo_sorted_compr_after,
                label="Fano Bound (%)",
                color=style_fano_after["color"],
                linestyle=style_fano_after["linestyle"],
                marker=style_fano_after["marker"],
                linewidth=2,
            )
            handles_cafter.append(h_fano)
            labels_cafter.append("Fano Bound (%)")
            ax1_compr_after.set_xlabel("Compression Ratio")
            ax1_compr_after.set_ylabel("Accuracy (%)")
            ax1_compr_after.set_title(
                f"{method_name}: Accuracy vs Compression (After Retrain)"
            )
            ax1_compr_after.grid(True)
            ax1_compr_after.legend(handles_cafter, labels_cafter, loc="best")
            fig_compr_after.tight_layout()
            fig_compr_after.savefig(
                os.path.join(
                    args.out_dir, f"accuracy_vs_compr_{method_name}_after.pdf"
                ),
                format="pdf",
            )
            plt.close(fig_compr_after)

        # --- The rest (per-class) unchanged ---
        # Keep the per-class plots as in the original, or add separate before/after if desired.

        per_class_binary_info_list_before = before_data.get("per_class_binary_info", [])
        per_class_binary_info_list_after = after_data.get("per_class_binary_info", [])
        accs_per_class_differentiate_dict_before = before_data.get(
            "accs_per_class_differentiate", {}
        )
        accs_per_class_differentiate_dict_after = after_data.get(
            "accs_per_class_differentiate", {}
        )

        # Handle both old format (list) and new format (dict by decoder)
        if isinstance(accs_per_class_differentiate_dict_before, list):
            accs_per_class_differentiate_dict_before = {
                "original": accs_per_class_differentiate_dict_before
            }
        if isinstance(accs_per_class_differentiate_dict_after, list):
            accs_per_class_differentiate_dict_after = {
                "original": accs_per_class_differentiate_dict_after
            }

        if (
            per_class_binary_info_list_before
            and per_class_binary_info_list_after
            and accs_per_class_differentiate_dict_before
            and accs_per_class_differentiate_dict_after
            and compr_list_before
            and compr_list_after
        ):
            first_info = per_class_binary_info_list_before[0]
            class_indices = sorted([int(k) for k in first_info.keys()])
            n_classes = len(class_indices)
            ncols = 5 if n_classes >= 5 else n_classes
            nrows = int(np.ceil(n_classes / ncols))

            compr_arr = np.array(compr_list_before[1:])

            collage_pdf_path = os.path.join(
                args.out_dir, f"per_class_fano_upper_{method_name}_collage.pdf"
            )
            pdf_pages = PdfPages(collage_pdf_path)
            fig_all, axes_all = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), squeeze=False
            )

            decoder_names = sorted(
                list(
                    set(
                        list(accs_per_class_differentiate_dict_before.keys())
                        + list(accs_per_class_differentiate_dict_after.keys())
                    )
                )
            )
            all_styles_perclass = get_distinct_styles(len(decoder_names) * 2 + 2)

            for i, class_idx in enumerate(class_indices):
                # [exactly as original for combined per-class plots...]
                i_x_w_k_list_before = []
                for per_class_binary_info in per_class_binary_info_list_before:
                    if str(class_idx) in per_class_binary_info:
                        i_x_w_k_list_before.append(
                            per_class_binary_info[str(class_idx)]["I_x_w_k"]
                        )
                    elif class_idx in per_class_binary_info:
                        i_x_w_k_list_before.append(
                            per_class_binary_info[class_idx]["I_x_w_k"]
                        )
                    else:
                        i_x_w_k_list_before.append(0.0)
                i_x_w_k_list_after = []
                for per_class_binary_info in per_class_binary_info_list_after:
                    if str(class_idx) in per_class_binary_info:
                        i_x_w_k_list_after.append(
                            per_class_binary_info[str(class_idx)]["I_x_w_k"]
                        )
                    elif class_idx in per_class_binary_info:
                        i_x_w_k_list_after.append(
                            per_class_binary_info[class_idx]["I_x_w_k"]
                        )
                    else:
                        i_x_w_k_list_after.append(0.0)

                i_x_w_k_arr_before = np.array(i_x_w_k_list_before[1:])
                i_x_w_k_arr_after = np.array(i_x_w_k_list_after[1:])

                balance = True
                ent = 1.0 if balance else -(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9))
                fano_acc_before = fano_upper_accuracy_from_I(
                    i_x_w_k_arr_before, K=2, H_Y_bits=ent
                )
                fano_acc_pct_before = 100.0 * np.array(fano_acc_before)

                fano_acc_after = fano_upper_accuracy_from_I(
                    i_x_w_k_arr_after, K=2, H_Y_bits=ent
                )
                fano_acc_pct_after = 100.0 * np.array(fano_acc_after)

                sort_idx = np.argsort(compr_arr)
                compr_sorted = compr_arr[sort_idx]
                fano_acc_sorted_before = fano_acc_pct_before[sort_idx]
                fano_acc_sorted_after = fano_acc_pct_after[sort_idx]

                # Extract per-class differentiate accuracies
                per_class_diff_by_decoder_before = {}
                for (
                    decoder_name,
                    accs_per_class_diff_list,
                ) in accs_per_class_differentiate_dict_before.items():
                    per_class_diff_list = []
                    for accs_per_class_diff in accs_per_class_diff_list:
                        if (
                            isinstance(accs_per_class_diff, list)
                            and len(accs_per_class_diff) > class_idx
                        ):
                            error_rate = accs_per_class_diff[class_idx]
                            if isinstance(error_rate, (list, np.ndarray)):
                                error_rate = (
                                    float(error_rate[0]) if len(error_rate) > 0 else 0.0
                                )
                            else:
                                error_rate = float(error_rate)
                            accuracy = 1.0 - error_rate
                            per_class_diff_list.append(accuracy)
                        else:
                            per_class_diff_list.append(0.0)
                    per_class_diff_arr = np.array(per_class_diff_list[1:])
                    per_class_diff_pct = 100.0 * per_class_diff_arr
                    per_class_diff_by_decoder_before[decoder_name] = per_class_diff_pct[
                        sort_idx
                    ]
                per_class_diff_by_decoder_after = {}
                for (
                    decoder_name,
                    accs_per_class_diff_list,
                ) in accs_per_class_differentiate_dict_after.items():
                    per_class_diff_list = []
                    for accs_per_class_diff in accs_per_class_diff_list:
                        if (
                            isinstance(accs_per_class_diff, list)
                            and len(accs_per_class_diff) > class_idx
                        ):
                            error_rate = accs_per_class_diff[class_idx]
                            if isinstance(error_rate, (list, np.ndarray)):
                                error_rate = (
                                    float(error_rate[0]) if len(error_rate) > 0 else 0.0
                                )
                            else:
                                error_rate = float(error_rate)
                            accuracy = 1.0 - error_rate
                            per_class_diff_list.append(accuracy)
                        else:
                            per_class_diff_list.append(0.0)
                    per_class_diff_arr = np.array(per_class_diff_list[1:])
                    per_class_diff_pct = 100.0 * per_class_diff_arr
                    per_class_diff_by_decoder_after[decoder_name] = per_class_diff_pct[
                        sort_idx
                    ]

                # Plot on subplot
                row = i // ncols
                col = i % ncols
                ax = axes_all[row][col]
                fano_style_before = all_styles_perclass[-2]
                fano_style_after = all_styles_perclass[-1]
                ax.plot(
                    compr_sorted,
                    fano_acc_sorted_before,
                    marker=fano_style_before["marker"],
                    label="Fano UB (Before)" if row == 0 and col == 0 else None,
                    color=fano_style_before["color"],
                    linestyle=fano_style_before["linestyle"],
                    linewidth=2,
                )
                ax.plot(
                    compr_sorted,
                    fano_acc_sorted_after,
                    marker=fano_style_after["marker"],
                    label="Fano UB (After)" if row == 0 and col == 0 else None,
                    color=fano_style_after["color"],
                    linestyle=fano_style_after["linestyle"],
                    linewidth=2,
                )
                for j, decoder_name in enumerate(decoder_names):
                    style_before = (
                        all_styles_perclass[2 * j]
                        if 2 * j < len(all_styles_perclass)
                        else {"color": "black", "linestyle": "-", "marker": "o"}
                    )
                    style_after = (
                        all_styles_perclass[2 * j + 1]
                        if 2 * j + 1 < len(all_styles_perclass)
                        else {"color": "gray", "linestyle": "--", "marker": "s"}
                    )
                    if decoder_name in per_class_diff_by_decoder_before:
                        ax.plot(
                            compr_sorted,
                            per_class_diff_by_decoder_before[decoder_name],
                            marker=style_before["marker"],
                            label=(
                                f"PC Diff ({decoder_name}) Before"
                                if row == 0 and col == 0
                                else None
                            ),
                            color=style_before["color"],
                            linestyle=style_before["linestyle"],
                            linewidth=1.5,
                            alpha=0.7,
                        )
                    if decoder_name in per_class_diff_by_decoder_after:
                        ax.plot(
                            compr_sorted,
                            per_class_diff_by_decoder_after[decoder_name],
                            marker=style_after["marker"],
                            label=(
                                f"PC Diff ({decoder_name}) After"
                                if row == 0 and col == 0
                                else None
                            ),
                            color=style_after["color"],
                            linestyle=style_after["linestyle"],
                            linewidth=1.5,
                            alpha=0.7,
                        )

                ax.set_xlabel("Compr. Ratio")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"Class {class_idx}", fontsize=11)
                ax.grid(True, alpha=0.3)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, loc="best")

            for j in range(n_classes, nrows * ncols):
                fig_all.delaxes(axes_all[j // ncols][j % ncols])

            plt.suptitle(
                f"{method_name}: Fano UB vs Per-Class Differentiate Acc. by Compression (Before/After Retrain)",
                fontsize=16,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf_pages.savefig(fig_all)
            plt.close(fig_all)

            # (Individual per-class pages unchanged)
            for i, class_idx in enumerate(class_indices):
                i_x_w_k_list_before = []
                for per_class_binary_info in per_class_binary_info_list_before:
                    if str(class_idx) in per_class_binary_info:
                        i_x_w_k_list_before.append(
                            per_class_binary_info[str(class_idx)]["I_x_w_k"]
                        )
                    elif class_idx in per_class_binary_info:
                        i_x_w_k_list_before.append(
                            per_class_binary_info[class_idx]["I_x_w_k"]
                        )
                    else:
                        i_x_w_k_list_before.append(0.0)

                i_x_w_k_list_after = []
                for per_class_binary_info in per_class_binary_info_list_after:
                    if str(class_idx) in per_class_binary_info:
                        i_x_w_k_list_after.append(
                            per_class_binary_info[str(class_idx)]["I_x_w_k"]
                        )
                    elif class_idx in per_class_binary_info:
                        i_x_w_k_list_after.append(
                            per_class_binary_info[class_idx]["I_x_w_k"]
                        )
                    else:
                        i_x_w_k_list_after.append(0.0)

                i_x_w_k_arr_before = np.array(i_x_w_k_list_before[1:])
                i_x_w_k_arr_after = np.array(i_x_w_k_list_after[1:])

                balance = True
                ent = 1.0 if balance else -(0.1 * math.log2(0.1) + 0.9 * math.log2(0.9))
                fano_acc_before = fano_upper_accuracy_from_I(
                    i_x_w_k_arr_before, K=2, H_Y_bits=ent
                )
                fano_acc_pct_before = 100.0 * np.array(fano_acc_before)

                fano_acc_after = fano_upper_accuracy_from_I(
                    i_x_w_k_arr_after, K=2, H_Y_bits=ent
                )
                fano_acc_pct_after = 100.0 * np.array(fano_acc_after)

                sort_idx = np.argsort(compr_arr)
                compr_sorted = compr_arr[sort_idx]
                fano_acc_sorted_before = fano_acc_pct_before[sort_idx]
                fano_acc_sorted_after = fano_acc_pct_after[sort_idx]

                per_class_diff_by_decoder_before = {}
                for (
                    decoder_name,
                    accs_per_class_diff_list,
                ) in accs_per_class_differentiate_dict_before.items():
                    per_class_diff_list = []
                    for accs_per_class_diff in accs_per_class_diff_list:
                        if (
                            isinstance(accs_per_class_diff, list)
                            and len(accs_per_class_diff) > class_idx
                        ):
                            error_rate = accs_per_class_diff[class_idx]
                            if isinstance(error_rate, (list, np.ndarray)):
                                error_rate = (
                                    float(error_rate[0]) if len(error_rate) > 0 else 0.0
                                )
                            else:
                                error_rate = float(error_rate)
                            accuracy = 1.0 - error_rate
                            per_class_diff_list.append(accuracy)
                        else:
                            per_class_diff_list.append(0.0)
                    per_class_diff_arr = np.array(per_class_diff_list[1:])
                    per_class_diff_pct = 100.0 * per_class_diff_arr
                    per_class_diff_by_decoder_before[decoder_name] = per_class_diff_pct[
                        sort_idx
                    ]

                per_class_diff_by_decoder_after = {}
                for (
                    decoder_name,
                    accs_per_class_diff_list,
                ) in accs_per_class_differentiate_dict_after.items():
                    per_class_diff_list = []
                    for accs_per_class_diff in accs_per_class_diff_list:
                        if (
                            isinstance(accs_per_class_diff, list)
                            and len(accs_per_class_diff) > class_idx
                        ):
                            error_rate = accs_per_class_diff[class_idx]
                            if isinstance(error_rate, (list, np.ndarray)):
                                error_rate = (
                                    float(error_rate[0]) if len(error_rate) > 0 else 0.0
                                )
                            else:
                                error_rate = float(error_rate)
                            accuracy = 1.0 - error_rate
                            per_class_diff_list.append(accuracy)
                        else:
                            per_class_diff_list.append(0.0)
                    per_class_diff_arr = np.array(per_class_diff_list[1:])
                    per_class_diff_pct = 100.0 * per_class_diff_arr
                    per_class_diff_by_decoder_after[decoder_name] = per_class_diff_pct[
                        sort_idx
                    ]

                decoder_names_single = sorted(
                    list(
                        set(
                            list(per_class_diff_by_decoder_before.keys())
                            + list(per_class_diff_by_decoder_after.keys())
                        )
                    )
                )
                styles_single = get_distinct_styles(len(decoder_names_single) * 2 + 2)

                fig, ax1 = plt.subplots(figsize=(7, 5))
                fano_style_before = styles_single[-2]
                fano_style_after = styles_single[-1]
                ax1.plot(
                    compr_sorted,
                    fano_acc_sorted_before,
                    marker=fano_style_before["marker"],
                    label="Fano UB (k=2) (Before Retrain) (%)",
                    color=fano_style_before["color"],
                    linestyle=fano_style_before["linestyle"],
                    linewidth=2,
                )
                ax1.plot(
                    compr_sorted,
                    fano_acc_sorted_after,
                    marker=fano_style_after["marker"],
                    label="Fano UB (k=2) (After Retrain) (%)",
                    color=fano_style_after["color"],
                    linestyle=fano_style_after["linestyle"],
                    linewidth=2,
                )
                for j, decoder_name in enumerate(decoder_names_single):
                    style_before = (
                        styles_single[2 * j]
                        if 2 * j < len(styles_single)
                        else {"color": "black", "linestyle": "-", "marker": "o"}
                    )
                    style_after = (
                        styles_single[2 * j + 1]
                        if 2 * j + 1 < len(styles_single)
                        else {"color": "gray", "linestyle": "--", "marker": "s"}
                    )
                    if decoder_name in per_class_diff_by_decoder_before:
                        ax1.plot(
                            compr_sorted,
                            per_class_diff_by_decoder_before[decoder_name],
                            marker=style_before["marker"],
                            label=f"PC Diff Acc ({decoder_name}) (Before) (%)",
                            color=style_before["color"],
                            linestyle=style_before["linestyle"],
                            linewidth=2,
                        )
                    if decoder_name in per_class_diff_by_decoder_after:
                        ax1.plot(
                            compr_sorted,
                            per_class_diff_by_decoder_after[decoder_name],
                            marker=style_after["marker"],
                            label=f"PC Diff Acc ({decoder_name}) (After) (%)",
                            color=style_after["color"],
                            linestyle=style_after["linestyle"],
                            linewidth=2,
                        )

                ax1.set_xlabel("Compression Ratio")
                ax1.set_ylabel("Accuracy (%)")
                ax1.set_title(
                    f"{method_name}: Class {class_idx} - Fano Bound & Per-Class Acc vs Compression (Before/After Retrain)",
                    fontsize=12,
                )
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc="best", fontsize=9)
                fig.tight_layout()
                pdf_pages.savefig(fig)
                plt.close(fig)

            pdf_pages.close()
            print(f"Saved collective collage PDF: {collage_pdf_path}")


if __name__ == "__main__":
    main()
