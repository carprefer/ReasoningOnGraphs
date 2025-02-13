import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from common.utils import *

class RoG():
    def __init__(self, args, device):
        self.maxNewTokens = args.maxNewTokens
        self.maxPlanTokens = args.maxPlanTokens
        self.maxTokenLength = args.maxTokenLength
        self.numBeam = args.numBeam

        self.tokenizer = AutoTokenizer.from_pretrained(args.rogModelName, padding_side='left', use_fast=False)
        if args.model == 'myRoG':
            {}
        self.model = AutoModelForCausalLM.from_pretrained(args.rogModelName, torch_dtype=torch.float16)
        self.model.to(device)

        #self.tokenizer.pad_token_id = self.model.config.eos_token_id

    def isPromptFit(self, prompt):
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokens) <= self.maxTokenLength
    
    def planning(self, prompts: list[str]):
        batchSize = len(prompts)
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
        inputIds = inputs['input_ids'].to(self.model.device)
        outputs = self.model.generate(
            input_ids=inputIds,
            attention_mask=inputs['attention_mask'].to(self.model.device),
            max_new_tokens=self.maxPlanTokens, 
            num_beams=self.numBeam,
            num_return_sequences=self.numBeam,
            output_scores=True,
            return_dict_in_generate=True,
            early_stopping=False,
            return_legacy_cache=True,
        )
        paths = self.tokenizer.batch_decode(outputs.sequences[:,inputIds.shape[-1]:], skip_special_tokens=True)
        paths = [p.strip() for p in paths]
        scores = outputs.sequences_scores.tolist()
        normScores = torch.softmax(outputs.sequences_scores, dim=0).tolist()
        results = []
        for b in range(batchSize):
            s = b * self.numBeam
            e = s + self.numBeam
            results.append({'paths': paths[s:e], 'scores': scores[s:e], 'normScores': normScores[s:e]})
        
        return results
    
    def retrieving(self, graph, plans, qEntities):
        reasoningPaths = []
        for qEntity in qEntities:
            for plan in plans:
                reasoningPath = bfs_with_rule(graph, qEntity, plan)
                reasoningPaths.extend(reasoningPath)
        
        return reasoningPaths

    
    def inference(self, prompts: list[str]):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
        inputIds = inputs['input_ids'].to(self.model.device)
        outputs = self.model.generate(
            input_ids=inputIds,
            attention_mask=inputs['attention_mask'].to(self.model.device),
            max_new_tokens=self.maxNewTokens, 
        )
        return self.tokenizer.batch_decode(outputs[:,inputIds.shape[-1]:], skip_special_tokens=True)
    
        

    
if __name__ == '__main__':
    {}

