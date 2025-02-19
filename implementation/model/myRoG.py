import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import AutoPeftModelForCausalLM

class MyRoG():
    def __init__(self, args, device):
        self.maxNewTokens = args.maxNewTokens
        self.maxPlanTokens = args.maxPlanTokens
        self.maxTokenLength = args.maxTokenLength
        self.numBeam = args.numBeam

        self.tokenizer = AutoTokenizer.from_pretrained(args.rogModelPath, padding_side='left', use_fast=False)
        self.model = AutoPeftModelForCausalLM.from_pretrained(args.rogModelPath, torch_dtype=torch.bfloat16)
        self.model.to(device)


        self.model.eval()

        #self.tokenizer.pad_token_id = self.model.config.eos_token_id
    
    def planning(self, prompts: list[str]):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
        inputIds = inputs['input_ids'].to(self.model.device)
        attentionMask = inputs['attention_mask'].to(self.model.device)
        outputs = self.model.generate(
            input_ids=inputIds,
            attention_mask=attentionMask,
            max_new_tokens=self.maxPlanTokens, 
            num_beams=self.numBeam,
            num_return_sequences=self.numBeam,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # remove question part & decode
        paths = self.tokenizer.batch_decode(outputs.sequences[:,inputIds.shape[-1]:], skip_special_tokens=True)
        paths = [p.strip() for p in paths]
        scores = outputs.sequences_scores.tolist()
        normScores = torch.softmax(outputs.sequences_scores, dim=0).tolist()
        results = []
        for b in range(len(prompts)):
            s = b * self.numBeam
            e = s + self.numBeam
            results.append({'paths': paths[s:e], 'scores': scores[s:e], 'normScores': normScores[s:e]})
        
        return results


    def inference(self, prompts: list[str]):
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
        inputIds = inputs['input_ids'].to(self.model.device)
        attentionMask = inputs['attention_mask'].to(self.model.device)
        outputs = self.model.generate(
            input_ids=inputIds,
            attention_mask=attentionMask,
            max_new_tokens=self.maxNewTokens, 
        )
        # remove question part & decode
        return self.tokenizer.batch_decode(outputs[:,inputIds.shape[-1]:], skip_special_tokens=True)

if __name__ == '__main__':
    {}

