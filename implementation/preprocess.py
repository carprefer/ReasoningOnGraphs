import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import json
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.cwqDataset import CwqDataset
from dataset.webQspDataset import WebQspDataset
from common.utils import *
from common.config import argparser
from common.prompter import Prompter


DATASET = {
    'cwq': CwqDataset,
    'webQsp': WebQspDataset,
}

def main(args):
    planningFilePath = f"../data/{args.model}/{args.dataset}/train_plans.jsonl"
    reasoningFilePath = f"../data/{args.model}/{args.dataset}/train_reasoning.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.modelName, padding_side='left', use_fast=False)

    print("Loading dataset ... ")
    dataset = DATASET[args.dataset]()
    trainset = list(dataset.dataset['train'])
    prompter = Prompter(tokenizer, args.maxTokenLength)


    print("Preprocessing planning train set ...")
    relationPathsList = []
    with open(planningFilePath, 'w') as f:
        planPrompts = prompter.generatePrompts(trainset, 'plan')
        for data, planPrompt in tqdm(zip(trainset, planPrompts), total=len(trainset)):
            relationPaths = getRelationPaths(data['q_entity'], data['a_entity'], data['graph'])
            planAnswers = encodeRelationPaths(relationPaths)
            for planAnswer in planAnswers:
                f.write(json.dumps({'text': planPrompt + ' ' + planAnswer + '</s>'}) + '\n')
            relationPathsList.append(relationPaths)

            reasoningPaths = []
            for qEntity in data['q_entity']: 
                for p in relationPaths:
                    reasoningPath = retrieveReasoningPathsFromRelationPath(data['graph'], qEntity, p)
                    reasoningPaths.extend(reasoningPath)
            data['reasoningPaths'] = reasoningPaths

    print("Preprocessing reasoning train set ...")
    with open(reasoningFilePath, 'w') as f:
        reasoningPrompts = prompter.generatePrompts(trainset, 'roG')
        for data, reasoningPrompt in tqdm(zip(trainset, reasoningPrompts), total=len(trainset)):
            relationPaths = getRelationPaths(data['q_entity'], data['a_entity'], data['graph'])
            f.write(json.dumps({'ground_paths': relationPaths, 'text': reasoningPrompt + ' ' + '\n'.join(data['answer']) + '</s>'}) + '\n')




if __name__ == "__main__":
    args = argparser.parse_args()
    main(args)
