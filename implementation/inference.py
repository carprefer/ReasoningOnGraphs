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
from model.myRoG import MyRoG
from dataset.cwqDataset import CwqDataset
from dataset.webQspDataset import WebQspDataset
from evaluate import eval_result
from common.utils import *
from common.config import argparser
from common.prompter import Prompter

MODEL = {
    'llm': Llm,
    'roG': RoG,
    'myRoG': MyRoG,
}

DATASET = {
    'cwq': CwqDataset,
    'webQsp': WebQspDataset,
}

def makePlans(accelerator, model, prompter, testBatchs, planPath):
    results = []
    for inputs in tqdm(testBatchs):
        with torch.no_grad():
            torch.cuda.empty_cache()
            outputs = model.planning(prompter.generatePrompts(inputs, 'plan'))
        for input, output in zip(inputs, outputs):
            result = {
                'id': input['id'],
                'question': input['question'],
                'prediction': parseRelationPaths(output['paths']),
                'ground_paths': getRelationPaths(input["q_entity"], input["a_entity"], input["graph"]),
                'raw_output': output,
            }
            results.append(result)

    gatheredResults = gather_object(results)
    if accelerator.is_main_process:
        with open(planPath, 'w') as f:
            for result in flatten(gatheredResults):
                f.write(json.dumps(result) + '\n')

# update testBatchs
def retrieveReasoningPaths(accelerator, testBatchs, planPath):
    plansets = splitDataset(loadJsonl(planPath), 0, accelerator.num_processes)
    planBatchs = makeBatchs(plansets[accelerator.process_index], len(testBatchs[0]))

    for datas, plans in tqdm(zip(testBatchs, planBatchs), total=len(testBatchs)):
        for data, plan in zip(datas, plans):
            reasoningPaths = []
            for qEntity in data['q_entity']: 
                for p in plan['prediction']:
                    reasoningPath = retrieveReasoningPathsFromRelationPath(data['graph'], qEntity, p)
                    reasoningPaths.extend(reasoningPath)
            data['reasoningPaths'] = reasoningPaths


def main(args):
    accelerator = Accelerator()
    outputPath = f"../output/{args.model}/{args.dataset}/predictions.jsonl"
    planPath = f"../data/{args.model}/{args.dataset}/plans.jsonl"

    accelerator.print("Loading dataset ... ")
    dataset = DATASET[args.dataset]()

    accelerator.print("Making testset ...")
    testsets = splitDataset(dataset.dataset['test'], args.testNum, accelerator.num_processes)
    testBatchs = makeBatchs(testsets[accelerator.process_index], args.testBatchSize)

    accelerator.print("Loading model ... ")
    model = MODEL[args.model](args, accelerator.device)
    prompter = Prompter(model.tokenizer, args.maxTokenLength)

    if args.model != 'llm':
        if not os.path.exists(planPath):
            accelerator.print("Making plans ...")
            makePlans(accelerator, model, prompter, testBatchs, planPath)
        accelerator.wait_for_everyone()
        accelerator.print("Retrieving reasoning paths ...")
        retrieveReasoningPaths(accelerator, testBatchs, planPath)


    accelerator.print("Inferencing ... ")

    results = []
    for inputs in tqdm(testBatchs):
        with torch.no_grad():
            torch.cuda.empty_cache()
            prompts = prompter.generatePrompts(inputs, args.model)
            outputs = model.inference(prompts)
        for input, output, prompt in zip(inputs, outputs, prompts):
            result = {
                'id': input['id'],
                'question': input['question'],
                'prediction': output,
                'ground_truth': input['answer'],
                'input': prompt,
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
