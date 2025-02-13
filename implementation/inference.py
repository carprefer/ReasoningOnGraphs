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
from common.prompter import generatePrompts

MODEL = {
    'llm': Llm,
    'roG': RoG,
}

DATASET = {
    'cwq': CwqDataset,
    'webQsp': WebQspDataset,
}

def main(args):
    accelerator = Accelerator()
    outputPath = f"../output/{args.model}/{args.dataset}/predictions.jsonl"
    planPath = f"../data/{args.model}/{args.dataset}/plans.jsonl"

    accelerator.print("Loading dataset ... ")
    dataset = DATASET[args.dataset]()
    accelerator.print("Making testset ...")
    testsets = splitDataset(dataset.dataset['test'], args.testNum, accelerator.num_processes)

    accelerator.print(f"Load model on {accelerator.process_index}... ")
    model = MODEL[args.model](args, accelerator.device)

    testset = testsets[accelerator.process_index]

    testBatchs = makeBatchs(testset, args.testBatchSize)

    accelerator.print(f"Inference on {accelerator.process_index}... ")
    if args.model == 'roG':
        model.plan()
    results = []
    for inputs in tqdm(testBatchs):
        with torch.no_grad():
            if args.model == 'roG':
                outputs = model.inference(generatePrompts(inputs, 'rogPlan'))
            else:
                outputs = model.inference(generatePrompts(inputs, 'llm'))
        for input, output in zip(inputs, outputs):
            if args.model == 'roG':
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
            else:
                result = {
                    'id': input['id'],
                    'question': input['question'],
                    'prediction': output,
                    'ground_truth': input['answer'],
                }
            results.append(result)

    gatheredResults = gather_object(results)
    
    if accelerator.is_main_process:
        if args.model == 'roG':
            outputPath = planPath
        with open(outputPath, 'w') as f:
            for result in flatten(gatheredResults):
                f.write(json.dumps(result) + '\n')
        if args.mode == 'llm':
            print("Evaluating ...")
            eval_result(outputPath)
    








if __name__ == "__main__":
    args = argparser.parse_args()
    torch.distributed.init_process_group(backend='nccl')
    main(args)
    torch.distributed.destroy_process_group()
