import re
import json
import random
import numpy as np
import networkx as nx
from collections import deque
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset

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

def smart_tokenizer_and_embedding_resize(
    new_tokens: list[str],
    special_tokens_dict: dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(new_tokens)
    num_new_special_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    total_new_tokens = num_new_tokens + num_new_special_tokens
    if total_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_multiple_datasets(data_path_list, shuffle=False):
    '''
    Load multiple datasets from different paths.

    Args:
        data_path_list (_type_): _description_
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    '''
    dataset_list = [load_dataset('json', data_files=p, split="train")
                     for p in data_path_list]
    dataset = concatenate_datasets(dataset_list)
    if shuffle:
        dataset = dataset.shuffle()
    return dataset