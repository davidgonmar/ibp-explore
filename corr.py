import argparse
import json
import math
import numpy as np


def binary_entropy_bits(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def fano_error_upper_bound(I_bits, K=10, H_Y_bits=None, tol=1e-7):
    if H_Y_bits is None:
        H_Y_bits = math.log2(K)
    target = H_Y_bits - I_bits
    if target <= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0 - 1.0 / K

    def g(p):
        return binary_entropy_bits(p) + p * math.log2(K - 1) - target

    if g(hi) < 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = g(mid)
        if val > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def fano_upper_accuracy_from_I(I_bits_values, K=10, H_Y_bits=None):
    accs = []
    for I_bits in I_bits_values:
        Pe = fano_error_upper_bound(I_bits, K=K, H_Y_bits=H_Y_bits)
        accs.append(1.0 - Pe)
    return np.array(accs, dtype=float)


def pearson_r(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("size mismatch")
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def corr_pruning(prune_results):
    decoder_set = set()
    strategy_rows = []
    pooled = {}
    for strat_name, strat_info in prune_results.items():
        theo = fano_upper_accuracy_from_I(strat_info["izy"])
        row = {}
        for dec_name, acc_list in strat_info["accs"].items():
            acc_arr = np.asarray(acc_list, dtype=float)
            r = pearson_r(theo, acc_arr)
            row[dec_name] = r
            decoder_set.add(dec_name)
            if dec_name not in pooled:
                pooled[dec_name] = {"theo": [], "acc": []}
            pooled[dec_name]["theo"].extend(theo.tolist())
            pooled[dec_name]["acc"].extend(acc_arr.tolist())
        strategy_rows.append((strat_name, row))
    pooled_row = {}
    for dec_name, both in pooled.items():
        pooled_row[dec_name] = pearson_r(both["theo"], both["acc"])
    strategy_rows.append(("ALL", pooled_row))
    decoder_names = sorted(list(decoder_set))
    return strategy_rows, decoder_names


def escape_latex(s):
    return s.replace("_", "\\_")


def latex_table_pruning(strategy_rows, decoder_names):
    cols = "l" + "c" * len(decoder_names)
    lines = []
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\hline")
    header = "Strategy"
    for dec in decoder_names:
        header += " & " + escape_latex(dec)
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for strat_name, row in strategy_rows:
        line = escape_latex(strat_name)
        for dec in decoder_names:
            r = row.get(dec, float("nan"))
            cell = "--" if math.isnan(r) else f"{r:.4f}"
            line += " & " + cell
        line += " \\\\"
        lines.append(line)
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def corr_quant(quant_results):
    theo = fano_upper_accuracy_from_I(quant_results["izy"])
    rows = {}
    for dec_name, acc_list in quant_results["accs"].items():
        rows[dec_name] = pearson_r(theo, acc_list)
    return rows


def latex_table_quant(rows):
    decoders = sorted(rows.keys())
    lines = []
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\hline")
    lines.append("Decoder & r \\\\")
    lines.append("\\hline")
    for dec in decoders:
        r = rows[dec]
        cell = "--" if math.isnan(r) else f"{r:.4f}"
        lines.append(escape_latex(dec) + " & " + cell + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def corr_factorization(factor_results):
    methods_dict = factor_results.get("methods", factor_results)
    decoder_set = set()
    method_rows = []
    pooled = {}
    for method_name, method_info in methods_dict.items():
        theo = fano_upper_accuracy_from_I(method_info["izy"])
        row = {}
        for dec_name, acc_list in method_info["accs"].items():
            acc_arr = np.asarray(acc_list, dtype=float)
            r = pearson_r(theo, acc_arr)
            row[dec_name] = r
            decoder_set.add(dec_name)
            if dec_name not in pooled:
                pooled[dec_name] = {"theo": [], "acc": []}
            pooled[dec_name]["theo"].extend(theo.tolist())
            pooled[dec_name]["acc"].extend(acc_arr.tolist())
        method_rows.append((method_name, row))
    pooled_row = {}
    for dec_name, both in pooled.items():
        pooled_row[dec_name] = pearson_r(both["theo"], both["acc"])
    method_rows.append(("ALL", pooled_row))
    decoder_names = sorted(list(decoder_set))
    return method_rows, decoder_names


def latex_table_factorization(method_rows, decoder_names):
    cols = "l" + "c" * len(decoder_names)
    lines = []
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\hline")
    header = "Method"
    for dec in decoder_names:
        header += " & " + escape_latex(dec)
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for method_name, row in method_rows:
        line = escape_latex(method_name)
        for dec in decoder_names:
            r = row.get(dec, float("nan"))
            cell = "--" if math.isnan(r) else f"{r:.4f}"
            line += " & " + cell
        line += " \\\\"
        lines.append(line)
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prune_json", type=str, default="results/pruning_all_layers.json"
    )
    parser.add_argument("--quant_json", type=str, default="results/quant_analysis.json")
    parser.add_argument(
        "--factor_json",
        type=str,
        default="results/factorization_analysis_resnet20.json",
    )
    args = parser.parse_args()
    prune_results = load_results(args.prune_json)
    quant_results = load_results(args.quant_json)
    factor_results = load_results(args.factor_json)
    strategy_rows, decoder_names = corr_pruning(prune_results)
    pruning_table = latex_table_pruning(strategy_rows, decoder_names)
    quant_rows = corr_quant(quant_results)
    quant_table = latex_table_quant(quant_rows)
    method_rows, decoder_names_factor = corr_factorization(factor_results)
    factor_table = latex_table_factorization(method_rows, decoder_names_factor)
    print(pruning_table)
    print()
    print(quant_table)
    print()
    print(factor_table)


if __name__ == "__main__":
    main()
