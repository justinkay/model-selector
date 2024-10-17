import matplotlib.pyplot as plt
from benchmark.bechmark_results import BenchmarkSummary, ExperimentResult
import numpy as np

from epsilon_optimization.gridsearch_results import GridsearchResult
from utils.math_utils import calculate_precentile_return_accuracy, smooth_data


def setup_toggle_function(fig, legend_lines, plot_lines):
    leg_to_lines = {}
    for legend, line in zip(legend_lines, plot_lines):
        legend.set_picker(5)
        leg_to_lines[legend] = line
    
    def on_pick(event):
        legend_line = event.artist

        ax_line = None
        if legend_line in leg_to_lines:
            ax_line = leg_to_lines[legend_line]
        else:
            return
        if isinstance(ax_line, tuple):
            visible = not ax_line[0].get_visible()
            for line in ax_line:
                line.set_visible(visible)
        else:
            visible = not ax_line.get_visible()
            ax_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick)


def plot_method_acc_diff(result: ExperimentResult, 
                            show=True,
                            threshold=None,
                            indices=None, 
                            smooth=None,
                            min_x=None,
                            max_x=None):
    
    num_samples = len(result.iteration_results["Random"]["bestModelAccuracyT"][0])
    x = np.arange(num_samples)[min_x:max_x]
    best_x_acc = calculate_precentile_return_accuracy(result, percentile=0.9, min_x=min_x, max_x=max_x)

    return plot_results(x, best_x_acc, show=show, threshold=threshold, indices=indices, smooth=smooth, y_title="Accuracy Difference")


def plot_method_success_prob(result: BenchmarkSummary, 
                                show=True,
                                threshold=None,
                                indices=None, 
                                smooth=None,
                                min_x=None,
                                max_x=None):
    x = np.arange(len(result.success_prob_mean["Random"])) if max_x is None else np.arange(max_x)
    if min_x is not None:
        x = x[min_x:]
    model_select_mean = {name: np.array(result.success_prob_mean[name]) for name in result.success_prob_mean.keys()}
    model_select_var = {name: np.array(result.success_prob_var[name]) for name in result.success_prob_var.keys()}

    return plot_results(x, model_select_mean, show=show, threshold=threshold, indices=indices, smooth=smooth, bounds=(0, 1.1, 0.1))


def plot_results(x, mean, var=None, 
                    show=True, 
                    threshold=None, 
                    indices=None, 
                    smooth=None,
                    y_title="Success Probability",
                    bounds=None):
    """
    Plots the performance of each method over all iterations of the benchmark.
    """
    fontsize = 14
    
    fig, ax = plt.subplots(figsize=(10, 6))

    lines_ax = []
    variances_ax = []
    if threshold is not None:    
        ax.axhline(y=threshold, color='r', linestyle='--')
    for i, name in enumerate(mean.keys()):
        if indices is not None and i not in indices:
            continue
        if smooth is not None:
            mean[name] = smooth_data(mean[name], kernel_size=smooth)
        mean_s = ax.plot(x, mean[name][:len(x)], label=name.replace("_", " "))[0]
        lines_ax.append(mean_s)
        if var is not None:
            var_s = ax.fill_between(x, mean[name]-var[name], mean[name]+var[name], alpha=0.2, label=f"{name} Var")
            variances_ax.append(var_s)


    ax.set_xticks(np.arange(x[0], len(x)+1, 100))
    if bounds is not None:
        ax.set_yticks(np.arange(bounds[0], bounds[1], bounds[2]))

    ax.set_ylabel(y_title, fontsize=fontsize)
    ax.set_xlabel('Num. Samples', fontsize=fontsize)
    
    leg = ax.legend(handles=(np.array(lines_ax) if var is not None else np.array(lines_ax)).tolist(), 
                     ncols=2 if len(lines_ax) > 6 else 1, 
                     fancybox=True, 
                     shadow=True, 
                     loc='lower right')
    ax.grid()
    setup_toggle_function(fig, leg.get_texts(), lines_ax+variances_ax)

    leg.set_draggable(True)
    fig.tight_layout()
    
    if show:
        plt.show()
    return fig


def plot_benchmark_summary(result: BenchmarkSummary, show=True, show_var=True, threshold=None, title=""):
    """
    Plots the performance of each method over all iterations of the benchmark.
    """
    fontsize = 14
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))  # 13, 8

    x = np.arange(len(result.success_prob_mean[list(result.success_prob_mean.keys())[0]]))
    model_select_mean = {name: np.array(result.success_prob_mean[name]) for name in result.success_prob_mean.keys()}
    model_select_var = {name: np.array(result.success_prob_var[name]) for name in result.success_prob_var.keys()}
    model_acc_mean = {name: np.array(result.selected_acc_mean[name]) for name in result.selected_acc_mean.keys()}
    model_acc_var = {name: np.array(result.selected_acc_var[name]) for name in result.selected_acc_var.keys()}

    cmap = plt.get_cmap('tab20')
    colors = cmap.colors  # Extract the colors from tab20
    
    lines_ax = []
    variances_ax = []
    if threshold is not None:    
        ax1.axhline(y=threshold, color='r', linestyle='--')
    for i, name in enumerate(result.success_prob_mean.keys()):
        mean_s = ax1.plot(x, model_select_mean[name], label=name, color=colors[i])[0]
        acc_mean = model_acc_mean[name]
        # mean_a = ax2.plot(x, acc_mean, label=name)[0]
        lines_ax.append((mean_s,))
        if show_var and False:
            var_s = ax1.fill_between(x, model_select_mean[name]-model_select_var[name], model_select_mean[name]+result.success_prob_var[name], alpha=0.2, label=f"{name} Var")
            var_a = ax2.fill_between(x, acc_mean-model_acc_var[name], acc_mean+model_acc_var[name], alpha=0.2, label=f"{name} Var")
            variances_ax.append((var_s, var_a))


    ax1.set_ylabel('Success Probability', fontsize=fontsize)
    # ax2.set_ylabel('Best Model Accuracy', fontsize=fontsize)
    ax1.set_xlabel('Num. Samples', fontsize=fontsize)
    
    leg = fig.legend(handles=(np.array(lines_ax)[:,0]).tolist(),     #handles=(np.concatenate([np.array(lines_ax)[:,0], np.array(variances_ax)[:,0]]) if show_var else np.array(lines_ax)[:,0]).tolist(), 
                     ncols=1, 
                     fancybox=True, 
                     shadow=True, 
                     loc='lower right')
    
    setup_toggle_function(fig, leg.get_texts(), lines_ax+variances_ax)

    plt.tight_layout()
    leg.set_draggable(True)
    
    plt.title(title)
    
    if show:
        plt.show()
    return fig


def plot_grid_search_results(result: ExperimentResult, epsilons: list, show=True):
    """
    Plots the performance of each method over all iterations of the benchmark.
    """
    avg_data = result.get_avg_over_iterations()
    x_label = ["{:.2f}".format(epsilon) for epsilon in epsilons]
    y_labels = ["Avg. Best Model Accuracy", "Avg. Best Model Selected"]
    data = [[],[]]
    
    for method in avg_data.model_ranking_t.keys():
        data[0].append(avg_data.avg_best_model_accuracy_t[method])
        data[1].append(avg_data.avg_best_model_selected_t[method])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
    d1 = np.array(data[0]).reshape((1,-1))
    d2 = np.array(data[1]).reshape((1,-1))
    
    im1 = ax1.imshow(d1, cmap='hot', interpolation='nearest')
    ax1.set_yticks([0], ['Avg. Best Model Accuracy'])
    ax1.set_xticks(np.arange(len(x_label)), x_label)

    im2 = ax2.imshow(d2, cmap='hot', interpolation='nearest')
    ax2.set_yticks([0], ['Avg. Best Model Selected'])
    ax2.set_xticks(np.arange(len(x_label)), x_label)

    # plt.imshow(data, cmap='hot', interpolation='nearest')
    fig.colorbar(im1, ax=ax1, orientation="horizontal")
    fig.colorbar(im2, ax=ax2, orientation="horizontal")
    fig.tight_layout()
    # plt.yticks(np.arange(len(y_labels)), y_labels)
    # plt.colorbar()
    if show:
        plt.show()
    return fig

def plot_optimal_epsilons(res: list[GridsearchResult], show=True):

    epsilons = res[0].epsilon_range
    fig = plt.figure(figsize=(5,5))
    for i, r in enumerate(res):
        if isinstance(r.epsilons, dict):    
            plt.scatter(i, r.epsilons["success_diff"]["optimal_epsilon"], label="Success Diff")
            plt.scatter(i, r.epsilons["fastest"]["optimal_epsilon"], label="Fastest")
            plt.scatter(i, r.epsilons["success_avg"]["optimal_epsilon"], label="Success Avg")
        else:
            plt.scatter(i, r.epsilons, label=r.method)
        print(f"{r.method}: {r.epsilons}")
    plt.xticks(np.arange(0,len(res),1), [r.method for r in res])
    plt.ylim(min(epsilons)-0.01, max(epsilons)+0.01)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()

    return fig