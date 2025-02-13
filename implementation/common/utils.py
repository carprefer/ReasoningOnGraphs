import re
import json
import random
import numpy as np
import networkx as nx
from collections import deque
from tqdm import tqdm

def loadJsonl(file_path):
    datas = []
    with open(file_path, 'r') as f:
        for line in f:
            datas.append(json.loads(line))
    return datas

def flatten(origin: list):
    flatted = []
    for o in origin:
        if isinstance(o, list):
            flatted += o
        else:
            flatted += [o]
    return flatted


def makeBatchs(dataList, batchSize):
    return [dataList[i:i+batchSize] for i in range(0, len(dataList), batchSize)]

def splitDataset(dataset, num, workerNum):
    idxs = range(len(dataset))
    if num != 0:
        idxs = random.sample(idxs, num)
    dataList = [dataset[i] for i in tqdm(idxs)]
    return [a.tolist() for a in np.array_split(dataList, workerNum)]

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths

def bfs_with_rule(graph, start_node, target_rule, max_p = 10):
    result_paths = []
    queue = deque([(start_node, [])])  
    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            # if len(result_paths) >= max_p:
            #     break
            
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel,neighbor)]))
    
    return result_paths

def path_to_string(path: list) -> str:
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
            
    return result.strip()

def parse_prediction(prediction):
    """
    Parse a list of predictions to a list of rules

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for p in prediction:
        path = re.search(r"<PATH>(.*)<\/PATH>", p)
        if path is None:
            continue
        path = path.group(1)
        path = path.split("<SEP>")
        if len(path) == 0:
            continue
        rules = []
        for rel in path:
            rel = rel.strip()
            if rel == "":
                continue
            rules.append(rel)
        results.append(rules)
    return results