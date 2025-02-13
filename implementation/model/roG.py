import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


class RoG():
    def __init__(self, args, device):
        self.maxNewTokens = args.maxNewTokens
        self.numBeam = args.numBeam
        self.mode = True

        self.tokenizer = AutoTokenizer.from_pretrained(args.rogModelName, use_fast=False)
        self.model = AutoPeftModelForCausalLM.from_pretrained(args.rogModelName, torch_dtype=torch.bfloat16)
        self.model.to(device)

        self.tokenizer.pad_token_id = self.model.config.eos_token_id

    def plan(self):
        self.mode = True
    
    def reason(self):
        self.mode = False
    
    def planning(self, prompts: list[str]):
        batchSize = len(prompts)
        inputIds = self.tokenizer.encode(prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False).to(self.model.device)
        outputs = self.model.generate(
            input_ids=inputIds, 
            max_new_tokens=self.maxNewTokens, 
            num_beams=self.numBeam,
            num_return_sequences=self.numBeam,
            output_scores=True,
            return_dict_in_generate=True,
        )
        paths = self.tokenizer.batch_decode(outputs.sequences[:, inputIds.shape[-1]:], skip_special_tokens=True)
        paths = [p.strip() for p in paths]
        scores = outputs.sequences_scores.tolist()
        normScores = torch.softmax(outputs.sequences_scores, dim=0).tolist()
        results = []
        for b in range(batchSize):
            s = b * batchSize 
            e = s + self.numBeam
            results.append({'paths': paths[s:e], 'scores': scores[s:e], 'normScores': normScores[s:e]})
        
        return results
    
    def inference(self, prompts: list[str]):
        if self.mode:
            self.planning(prompts)
        

    
if __name__ == '__main__':
    {}

