import re
import json
import random
import numpy as np
import networkx as nx
from collections import deque
from tqdm import tqdm

def loadJsonl(filePath: str) -> list:
    with open(filePath, 'r') as f:
        return [json.loads(line) for line in f]

def flatten(origin: list) -> list:
    flatted = []
    for o in origin:
        if isinstance(o, list):
            flatted += o
        else:
            flatted += [o]
    return flatted


def makeBatchs(dataList: list, batchSize: int) -> list:
    return [dataList[i:i+batchSize] for i in range(0, len(dataList), batchSize)]

def splitDataset(dataset, num: int, workerNum: int) -> list[list]:
    idxs = range(len(dataset))
    if num != 0:
        idxs = random.sample(idxs, num)
    dataList = [dataset[i] for i in tqdm(idxs)]
    return [a.tolist() for a in np.array_split(dataList, workerNum)]

def triples2graph(triples: list) -> nx.Graph:
    graph = nx.Graph()
    for src, edge, dst in triples:
        graph.add_edge(src.strip(), dst.strip(), relation=edge.strip())
    return graph

def path2string(path: list) -> str:
    if len(path) == 0:
        return ""
    
    return path[0][0] + "".join([f" -> {edge} -> {dst}" for src, edge, dst in path])

def getRelationPaths(qEntities: list, aEntities: list, triples: list) -> list:
    graph = triples2graph(triples)

    paths = []
    for src in qEntities:
        if src not in graph:
            continue
        for dst in aEntities:
            if dst not in graph:
                continue
            try:
                paths += list(nx.all_shortest_paths(graph, src, dst))
            except:
                pass
    # extract relation only & make them unique
    return list(set([tuple([graph[p[i]][p[i+1]]['relation'] for i in range(len(p)-1)]) for p in paths]))

def parseRelationPaths(pred: list[str]) -> list[list]:
    relationPaths = []
    for p in pred:
        path = re.search(r"<PATH>(.*)<\/PATH>", p)
        if path is None:
            continue
        path = path.group(1).split("<SEP>")
        if len(path) == 0:
            continue
        path = list(filter(lambda x: x != "", [r.strip() for r in path]))
        relationPaths.append(path)
    return relationPaths

def encodeRelationPaths(relationPaths: list) -> str:
    return ['<PATH>' + '<SEP>'.join(rp) + '</PATH>' for rp in relationPaths]

def retrieveReasoningPathsFromRelationPath(triples: list, startNode: str, relationPath: list) -> list[list]:
    graph = triples2graph(triples)
    reasoningPaths = []
    queue = deque([(startNode, [])])  
    while queue:
        curNode, curPath = queue.popleft()

        if len(curPath) == len(relationPath):
            reasoningPaths.append(curPath)

        elif len(curPath) < len(relationPath):
            if curNode not in graph:
                continue
            for neighbor in graph.neighbors(curNode):
                rel = graph[curNode][neighbor]['relation']
                if rel == relationPath[len(curPath)]:
                    queue.append((neighbor, curPath + [(curNode, rel, neighbor)]))
    
    return reasoningPaths