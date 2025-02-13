import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import json
import torch
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import gather_object

from model.llm import Llm
from model.roG import RoG
from dataset.cwqDataset import CwqDataset
from dataset.webQspDataset import WebQspDataset
from evaluate import eval_result
from common.utils import *
from common.config import argparser
from common.prompter import Prompter

MODEL = {
    'llm': Llm,
    'roG': RoG,
    'myRoG': RoG,
}

DATASET = {
    'cwq': CwqDataset,
    'webQsp': WebQspDataset,
}

def makePlans(accelerator, model, prompter, testBatchs, planPath):
    accelerator.print("Making plans ...")
    results = []
    for inputs in tqdm(testBatchs):
        with torch.no_grad():
            outputs = model.planning(prompter.generatePrompts(inputs, 'plan'))
        for input, output in zip(inputs, outputs):
            graph = build_graph(input["graph"])
            paths = get_truth_paths(input["q_entity"], input["a_entity"], graph)
            ground_paths = set()
            for path in paths:
                ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
            result = {
                'id': input['id'],
                'question': input['question'],
                'prediction': parse_prediction(output['paths']),
                'ground_paths': list(ground_paths),
                'raw_output': output,
            }
            results.append(result)

    gatheredResults = gather_object(results)
    if accelerator.is_main_process:
        with open(planPath, 'w') as f:
            for result in flatten(gatheredResults):
                f.write(json.dumps(result) + '\n')

# update testBatchs
def retrieveReasoningPaths(accelerator, model, testBatchs, planPath):
    plansets = splitDataset(loadJsonl(planPath), 0, accelerator.num_processes)
    planBatchs = makeBatchs(plansets[accelerator.process_index], len(testBatchs[0]))
    for datas, plans in tqdm(zip(testBatchs, planBatchs)):
        for data, plan in zip(datas, plans):
            data['reasoningPaths'] = model.retrieving(data['graph'], plan['prediction'], data['q_entity'])
    


def main(args):
    accelerator = Accelerator()
    outputPath = f"../output/{args.model}/{args.dataset}/predictions.jsonl"
    planPath = f"../data/{args.model}/{args.dataset}/plans.jsonl"

    accelerator.print("Loading dataset ... ")
    dataset = DATASET[args.dataset]()

    accelerator.print("Making testset ...")
    testsets = splitDataset(dataset.dataset['test'], args.testNum, accelerator.num_processes)

    accelerator.print("Loading model ... ")
    model = MODEL[args.model](args, accelerator.device)
    prompter = Prompter(model)

    testset = testsets[accelerator.process_index]

    testBatchs = makeBatchs(testset, args.testBatchSize)

    if args.model == 'roG':
        if not os.path.exists(planPath):
            makePlans(accelerator, model, prompter, testBatchs, planPath)
        accelerator.wait_for_everyone()
        accelerator.print("Retrieving reasoning paths ...")
        retrieveReasoningPaths(accelerator, model, testBatchs, planPath)

    accelerator.print("Inferencing ... ")

    results = []
    for inputs in tqdm(testBatchs):
        with torch.no_grad():
            outputs = model.inference(prompter.generatePrompts(inputs, args.model))
        for input, output in zip(inputs, outputs):
            result = {
                'id': input['id'],
                'question': input['question'],
                'prediction': output,
                'ground_truth': input['answer'],
            }
            results.append(result)

    gatheredResults = gather_object(results)
    
    if accelerator.is_main_process:
        with open(outputPath, 'w') as f:
            for result in flatten(gatheredResults):
                f.write(json.dumps(result) + '\n')
        print("Evaluating ...")
        eval_result(outputPath)
    








if __name__ == "__main__":
    args = argparser.parse_args()
    torch.distributed.init_process_group(backend='nccl')
    main(args)
    torch.distributed.destroy_process_group()
