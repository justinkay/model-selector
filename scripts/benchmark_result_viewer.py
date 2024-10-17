import argparse
import os
import sys

path = os.path.join(os.path.dirname(__file__),"../src")
sys.path.append(path)

from utils.math_utils import smooth_data
from utils.fs_utils import load_benchmark_summary, load_experiment_result
from utils.result_plotting import plot_benchmark_summary, plot_method_success_prob, plot_method_acc_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=False, help="Path to the benchmark summary file.")
    parser.add_argument("-o", "--output", type=str, help="Output filename for the plot.")
    parser.add_argument("-s", "--smooth", type=int, help="Kernel size for smoothing the data.")
    parser.add_argument("-v", "--show_var", action="store_true", help="Show variance in the plot.")
    parser.add_argument("-a", "--show_acc", action="store_true", help="Show the accuracy plot.")
    parser.add_argument("-t", "--threshold", type=float, help="Threshold for the success probability.")
    parser.add_argument("--max_x", type=int, help="Maximum x range to show.")
    parser.add_argument("--min_x", type=int, help="Minimum x range to show.")
    parser.add_argument('show_indices', metavar='N', type=int, nargs='*', help='Optional list of indices for methods to show')
    args = parser.parse_args()

    res = load_experiment_result(args.data)
    summary = res.get_benchmark_summary()

    if args.show_acc:
        fig = plot_method_acc_diff(res, show=True, 
                                    threshold=args.threshold,
                                    indices=args.show_indices if len(args.show_indices) > 0 else None,
                                    min_x=args.min_x,
                                    max_x=args.max_x,
                                    smooth=args.smooth)
    else:
        fig = plot_method_success_prob(summary, show=True,
                                        threshold=args.threshold,
                                        indices=args.show_indices if len(args.show_indices) > 0 else None,
                                        min_x=args.min_x,
                                        max_x=args.max_x,
                                        smooth=args.smooth)

    if args.output is not None:
        dirname = os.path.dirname(args.output)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(f"Created directory {dirname}!")
        fig.savefig(args.output)
        print(f"Saved plot to {args.output}!")

    
