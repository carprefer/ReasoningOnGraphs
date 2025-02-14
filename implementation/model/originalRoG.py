import torch
from transformers import pipeline
from common.utils import *


class OriginalRoG():
    def __init__(self, args, device):
        self.maxNewTokens = args.maxNewTokens
        self.maxTokenLength = args.maxTokenLength

        self.generator = pipeline(
            'text-generation',
            model=args.rogModelName,
            torch_dtype=torch.bfloat16,
            use_fast=False,
            device=device,
        )
        self.generator.tokenizer.padding_side = 'left'
        #self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id

    def isPromptFit(self, prompt):
        tokens = self.generator.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokens) <= self.maxTokenLength

    def retrieving(self, graph, plans, qEntities):
        reasoningPaths = []
        for qEntity in qEntities:
            for plan in plans:
                reasoningPath = bfs_with_rule(graph, qEntity, plan)
                reasoningPaths.extend(reasoningPath)
        
        return reasoningPaths
    
    def inference(self, prompts: list[str]):
        outputs = self.generator(prompts, return_full_text=False, max_new_tokens=self.maxNewTokens, batch_size=len(prompts), padding=True)
        return [o[0]['generated_text'] for o in outputs]

    
if __name__ == '__main__':
    {}

