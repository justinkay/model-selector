from enum import Enum, auto
from queue import PriorityQueue
from typing import Any
import numpy as np
from tqdm import tqdm
from model_selection.method_config import MethodConfig
from data_provider.interface_data_provider import DataProvider
from model_selection.methods.galaxy import ClassSelection, GALAXYConfig
from model_selection.model_selection_strategy import ModelSelectionStrategy
from models.interface_inferable import Inferable
from utils.math_utils import distribution


class Node:
    def __init__(self, idx, pred, label):
        self.idx = idx
        self.pred = pred
        self.labeled = False
        self.label = label

    def update(self, classes, y):
        assert not self.labeled
        self.labeled = True
        self.label = {c: y==c for c in classes}

class GALAXY(ModelSelectionStrategy):
    strategy_type ="galaxy_labelbench"

    def __init__(self, method_config: dict, data_provider: DataProvider) -> None:
        super().__init__(GALAXYConfig(method_config), data_provider)
        self.class_selection=self.method_config.class_selection
        self.init_samples=self.method_config.init_samples

    def __call__(self, data: np.ndarray, predictions: np.ndarray, budget: int) -> Any:
        Lt = []
        num_instances = predictions.shape[0]
        num_models = predictions.shape[1]
        num_classes = len(self.data_provider.get_classes())
        classes = np.arange(num_classes)
        class_selection = self._get_selection_strategy()

        # Initial uniform random label querying with size given as hyperparameter 
        indices = list(range(num_instances))
        for _ in range(self.init_samples):
            i = indices.pop(np.random.randint(0,len(indices)))
            x = data[i]
            y = self.data_provider.query_label(x)
            Lt.append((i,x,y))

        nodes = []
        nodes_map = {}
        preds = []
        for i in range(len(predictions)):
            preds.append(distribution(predictions[i], n=num_classes))
        preds = np.array(preds)
        
        most_confident = np.max(preds, axis=1).reshape((-1, 1))
        margins = preds - most_confident + 1e-8 * most_confident if num_classes > 2 else preds[:, 0:1]
        for idx, margin in enumerate(margins):
            nodes.append(Node(idx, margin, None))
            nodes_map[idx] = nodes[-1]
        for i,x,y in Lt:
            nodes_map[i].update(classes, y)

        graphs = []
        graphs_left = []
        graphs_right = []
        for i in range(num_classes if num_classes > 2 else 1):
            sort_idx = np.argsort(-margins[:, i])
            graphs.append([nodes[idx] for idx in sort_idx])
            graphs_left.append(graphs[-1][1:] + [graphs[-1][-1]])
            graphs_right.append(([graphs[-1][0]] + graphs[-1][:-1])[::-1])

        # Start collect examples and label.
        budget -= self.init_samples
        for t in range(budget):
            k = class_selection(t, num_classes)
            graph = graphs[k]
            graph_left = graphs_left[k]
            graph_right = graphs_right[k]
            last = None
            shortest_lst = None
            lst = None
            last_unlabel_left = None
            last_unlabel_right = None
            nearest_node = None
            nearest_dist = len(graph) + 1

            # Left to right.
            node = graph[0]
            for i, next_node_left in enumerate(graph_left):
                if node.labeled:
                    if (last is not None) and (lst is not None) and \
                            ((shortest_lst is None) or (shortest_lst[1] - shortest_lst[0] > lst[1] - lst[0])) and \
                            (node.label[k] != last.label[k]):
                        shortest_lst = lst
                    last = node
                    lst = None
                    if node.label[k] == 0 and next_node_left.labeled and next_node_left.label[k] == 1 and \
                            last_unlabel_left is not None and nearest_dist >= i - last_unlabel_left:
                        nearest_dist = i - last_unlabel_left
                        nearest_node = last_unlabel_left
                else:
                    if lst is None:
                        lst = [i, i]
                    lst[1] = i
                    last_unlabel_left = i
                node = next_node_left

            if shortest_lst is None:
                # Right to left.
                node = graph[-1]
                for i, next_node_right in enumerate(graph_right):
                    i_right = len(graph) - 1 - i
                    if node.labeled and next_node_right.labeled and node.label[k] == 1 and \
                            next_node_right.label[k] == 0 and last_unlabel_right is not None and \
                            nearest_dist > last_unlabel_right - i_right:
                        nearest_dist = last_unlabel_right - i_right
                        nearest_node = last_unlabel_right
                    elif not node.labeled:
                        last_unlabel_right = i_right
                    node = next_node_right

                # If one or two classes don't have labeled node, randomly choose one.
                if nearest_node is None:
                    perm = np.random.permutation(len(graph))
                    for i in perm:
                        if not graph[i].labeled:
                            nearest_node = i
                            break
                idx = graph[nearest_node].idx
                x = data[idx]
                y = self.data_provider.query_label(x)
                graph[nearest_node].update(classes, y)
            else:
                i = (shortest_lst[0] + shortest_lst[1]) // 2
                idx = graph[i].idx
                x = data[idx]
                y = self.data_provider.query_label(x)
                graph[i].update(classes, y)
            Lt.append((idx, x, y))
        return Lt

    def _get_selection_strategy(self):
        def random(t, num_classes):
            return np.random.randint(0, num_classes)
        def round_robin(t, num_classes):
            return t % num_classes
        if self.class_selection==ClassSelection.RANDOM:
            return random
        elif self.class_selection==ClassSelection.ROUND_ROBIN:
            return round_robin
        else:
            raise ValueError(f"Unkown class selection strategy selected: {self.class_selection}")
