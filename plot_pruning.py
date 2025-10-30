import argparse
from utils import load_results_json, plot_pruning_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    res = load_results_json(args.input_json)
    plot_pruning_results(res, args.out_dir)


if __name__ == "__main__":
    main()
