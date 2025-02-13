import torch
from transformers import pipeline


class Llm():
    def __init__(self, args, device):
        self.maxNewTokens = args.maxNewTokens

        self.generator = pipeline(
            'text-generation',
            model=args.modelName,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id
    
    def inference(self, prompts: list[str]):
        outputs = self.generator(prompts, return_full_text=False, max_new_tokens=self.maxNewTokens, batch_size=len(prompts), padding=True)
        return [o[0]['generated_text'] for o in outputs]

    
if __name__ == '__main__':
    {}

