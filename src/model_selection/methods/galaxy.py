from enum import Enum, auto
from queue import PriorityQueue
from typing import Any
import numpy as np
from model_selection.method_config import MethodConfig
from model_selection.model_selection_strategy import ModelSelectionStrategy
import copy

from utils.math_utils import distribution
class GALAXYConfig(MethodConfig):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.class_selection = ClassSelection[str(config["class_selection"]).upper()] if "class_selection" in config else ClassSelection.RANDOM
        self.init_samples = config["init_samples"] if "init_samples" in config else 16
        self.ordering = config["ordering"] if "ordering" in config else True
        self.all_graphs = config["all_graphs"] if "all_graphs" in config else False


class ClassSelection(Enum):
    RANDOM=auto()
    ROUND_ROBIN=auto()

class Node:
    def __init__(self, idx: int, loc: int, label: int = None):
        self.idx: int = idx # index in original dataset (reference to datapoint x)
        self.loc: int = loc # location (ranking according to mrgin) in graph
        self.label: int = label 

    def is_labeled(self):
        return self.label != None

class Graph:

    def __init__(self, nodes: list[Node], edges: dict, node_map: dict, c: int, positives: list[int], negatives: list[int]) -> None:
        self.c: int = c
        self.order = 1
        self.nodes: list[Node] = nodes
        self.edges: dict = edges # map from node to list of neighbors
        self.node_map: dict = node_map # map from index to node
        self.positives: list[Node] = positives # list of nodes in graph with label c
        self.negatives: list[Node] = negatives # list of nodes in graph with label not c

    def add_edges(self, n1: Node, ns: list[Node]):
        self.edges[n1] | set(ns)

    def set_edges(self, n1: Node, ns: list[Node]):
        self.edges[n1] = set(ns)

    def label_node(self, idx: int, y: int):
        node: Node = self.node_map[idx]
        node.label = self.c==y
        if node.label:
            self.positives.append(node)
        else:
            self.negatives.append(node)
        new_edges = []
        # If there is a neighbor has opposite label this direction can never produce a sample again -> Break the edge between node and the direction of the opposite label if it exists
        for neighbor in self.edges[node]:
            if neighbor.is_labeled() and neighbor.label != node.label:
                self.edges[neighbor].remove(node)
            else:
                new_edges.append(neighbor)
        self.set_edges(node, new_edges)

    def shortest_shortest_path(self, second_order: bool) -> list[Node]:
        if len(self.positives) == 0 or len(self.negatives) == 0:
            return None
        path = None
        queue = PriorityQueue()
        prev_node = {} # map from node to previous node
        dist = {} # map from node to distance            

        edges = self.edges
        if second_order:
            edges = self._higher_order_edges(order=2)

        # initialize distances and previous nodes
        cnt = 0
        for n in self.nodes:
            if n.label == True:
                prev_node[n] = None
                dist[n] = 0
                queue.put((0,cnt,n))    
                cnt += 1
            else:
                dist[n] = np.inf
                prev_node[n] = None
        
        while not queue.empty():
            _,_,node = queue.get()
            for neighbor in edges[node]:
                new_d = dist[node] + 1
                # if smaller distance to new node is found update distance and previous node and add to queue
                if new_d < dist[neighbor]:
                    dist[neighbor] = new_d
                    prev_node[neighbor] = node
                    queue.put((new_d, cnt, neighbor))
                    cnt += 1
                # since all edges have weight 1 we can break if we found a path (as we do breadth first search and the first path is the shortest one)
                if neighbor.label == False:
                    current = neighbor
                    path = [current]
                    # backtrace path
                    while prev_node[current] != None:
                        prev = prev_node[current]
                        path.append(prev)
                        current = prev
                    return path
        return path

    def _higher_order_edges(self, order=2):
        e = copy.copy(self.edges)
        for i in range(order):
            Graph._check_add_edge(e, self.nodes[i], self.nodes[i + order])
            Graph._check_add_edge(e, self.nodes[-i - 1], self.nodes[-i - order - 1])
        for i in range(order, len(self.nodes)-order):
            Graph._check_add_edge(e, self.nodes[i], self.nodes[i - order])
            Graph._check_add_edge(e, self.nodes[i], self.nodes[i + order])
        return e

    def _check_add_edge(edges: dict, n1: Node, n2: Node):
        if (n1.label == n2.label) or n1.label == None or n2.label == None:
            edges[n1].add(n2)
            edges[n2].add(n1)

class GALAXY(ModelSelectionStrategy):
    strategy_type ="galaxy"

    def __init__(self, method_config: dict) -> None:
        super().__init__(GALAXYConfig(method_config))
        self.class_selection=self.method_config.class_selection
        self.init_samples=self.method_config.init_samples
        self.ordering=self.method_config.ordering
        self.all_graphs = self.method_config.all_graphs

    def __call__(self, data: np.ndarray, predictions: np.ndarray, oracle: np.ndarray, classes: np.ndarray, budget: int) -> Any:
        Lt = []
        num_unlabeled = len(data)
        num_classes = len(classes)
        selection_strat: function = self._get_selection_strategy() # select a graph selection strategy based on hyperparameter

        # Initial uniform random label querying with size given as hyperparameter 
        indices = list(range(num_unlabeled))
        for _ in range(self.init_samples):
            i = indices.pop(np.random.randint(0,num_unlabeled))
            x = data[i]
            y = oracle[i]
            Lt.append((i,x,y))
            num_unlabeled -= 1

        margins, class_confidence = self._confidence_margins(predictions, classes) # NxC, NxC: Calculate confidence margin over all datapoints an classes 
        graphs = self._build_graph(margins, classes, Lt, order=1) # Generate linear graph for each class and sort based on prediction margins

        for t in range(budget-self.init_samples):
            if self.all_graphs:
                shortest_shortest_path = None
                shortest_path_length = np.inf
                for c, g in graphs.items():
                    # find shortest shortest path if possible and use it to select next sample
                    shortest_path = g.shortest_shortest_path(second_order=False)
                    if shortest_path != None and len(shortest_path) < shortest_path_length:
                        shortest_shortest_path = shortest_path
                        shortest_path_length = len(shortest_path)
                    
                if shortest_shortest_path == None or shortest_path_length < 3:    
                    for c, g in graphs.items():
                        # increase order of all graphs and try to calculate shortest path again
                        shortest_path = g.shortest_shortest_path(second_order=True)
                        if shortest_path != None and len(shortest_path) < shortest_path_length:
                            shortest_shortest_path = shortest_path
                            shortest_path_length = len(shortest_path)
            else:
                # select a graph based on strategy
                c = selection_strat(t, num_classes) 
                g: Graph = graphs[c]
                shortest_shortest_path = g.shortest_shortest_path(second_order=False)
                if shortest_shortest_path == None or len(shortest_shortest_path) < 3:    
                    shortest_shortest_path = g.shortest_shortest_path(second_order=True)

            if shortest_shortest_path == None or len(shortest_shortest_path) < 3: 
                # if no path is found or path is too short select random node
                xi = np.random.randint(0,num_unlabeled)
                idx = indices.pop(xi)
            else:
                # use shortest path and select the datapoint in the middle of it
                idx = self._sample_from_path(shortest_shortest_path)
                indices.remove(idx)

            # query label based on selected datapoint (random or middle of uncertainty zone given shortest path)
            x = data[idx] # get sample from initial data set
            y = oracle[idx]
            Lt.append((idx, x, y)) # node.idx is the reference to the original index in dataset
            num_unlabeled -= 1

            # update graphs by setting labels to the effected nodes and updating the important datastructures
            for graph in graphs.values():
                graph.label_node(idx, y)
        return Lt

    def _sample_from_path(self, path: list[Node]):
        l = len(path)
        if l % 2 == 0:
            node_i = l // 2 + np.random.randint(-1,1) # choose one of the 2 middle options
        else:
            node_i = l // 2
        node = path[node_i]
        idx = node.idx
        return idx

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

    def _confidence_margins(self, predictions, classes) -> tuple[np.ndarray, np.ndarray]:
        num_classes = len(classes)
        class_confidence = np.zeros((predictions.shape[0], num_classes)) # NxC: class confidence per datapoint
        margins = np.zeros((predictions.shape[0], num_classes)) # NxC: margins per class per datapoint
        
        for i in range(predictions.shape[0]):
            dist = distribution(predictions[i], n=num_classes)
            max_prob = max(dist)
            # calculate the class confidence by voting of experts converted to distribution
            class_confidence[i] = dist
            # calculate the margin by the difference of the highest and the class confidence and breaking ties with highest class probability
            margins[i] = dist - max_prob + 1e-8 * max_prob
        return margins, class_confidence
        
    def _build_graph(self, margins: np.ndarray, classes, init_labeled: list[tuple], order=1) -> dict:
        graphs = {}
        # convert init labels to mapping from datapoint indice to label
        label_i = {i: y for (i,x,y) in init_labeled}

        for c in range(len(classes)):
            # sort samples according to their likelihood of belonging to class c (highest margin first)
            if self.ordering:
                margins_c = np.argsort(margins[:,c]) # since we calculate class_prob - max_prob the highest margin is min (as its negative) -> argsort sorts ascending
            else:
                margins_c = np.argsort(-margins[:,c]) # since we calculate class_prob - max_prob the highest margin is min (as its negative) -> argsort sorts ascending
            nodes_c = []
            node_map_c = {}
            edges_c = {}
            p = []
            n = []
            # for each class c add all nodes sorted by their margin, mark nodes as labeled (label == c) when their datapoint (index: idx) was labeled in the initialization
            for i, idx in enumerate(margins_c):
                label = (c == label_i[idx]) if idx in label_i.keys() else None
                node = Node(idx, i, label)
                edges_c[node] = set()
                # if label equals graph label add node location to positives, else to negatives
                if label:
                    p.append(node)
                elif label != None and not label:
                    n.append(node)
                nodes_c.append(node) 
                node_map_c[idx] = node
            # # connect outside nodes according to order
            graphs[c] = Graph(nodes_c, edges_c, node_map_c, c, p, n)
            for i in range(order): 
                graphs[c].add_edges(nodes_c[i], nodes_c[i+1:i+order+1])
                graphs[c].add_edges(nodes_c[-i-1], nodes_c[-i-order-1:-i-1])
            # connect general nodes according to order
            for i in range(order, len(nodes_c)-order):
                graphs[c].add_edges(nodes_c[i], nodes_c[i+1:i+order+1])
                graphs[c].add_edges(nodes_c[i], nodes_c[i-order:i])

        return graphs