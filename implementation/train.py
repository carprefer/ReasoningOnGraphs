import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import AutoPeftModelForCausalLM, LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.trainer_utils import get_last_checkpoint
from dataclasses import dataclass, field

from common.utils import *
from common.config import argparser

@dataclass
class MyTrainingArguments(TrainingArguments):
    bf16: bool = field(default=False, metadata={"help": "Whether to use bf16 mixed precision training."})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length the model will process."})
    dataset_text_field: str = field(default="text", metadata={"help": "The name of the text field in the dataset."})

def main(args):
    #accelerator = Accelerator()
    trainPathList = [
        #f'../data/{args.model}/webQsp/test.jsonl',
        f'../data/{args.model}/webQsp/train_plans.jsonl',
        f'../data/{args.model}/webQsp/train_reasoning.jsonl',
        f'../data/{args.model}/cwq/train_plans.jsonl',
        f'../data/{args.model}/cwq/train_reasoning.jsonl',
    ]
    outputPath = f"../output/myRoG/trained_model2"
    logPath = f"../output/myRoG/logs2"

    model = AutoModelForCausalLM.from_pretrained(args.modelName, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    #model = DataParallel(model)
    model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.modelName, use_fast=False)
    smart_tokenizer_and_embedding_resize(
        ['<SEP>','<PATH>','</PATH>'],
        {'pad_token': '<PAD>'},
        tokenizer,
        model
    )
    tokenizer.padding_side='right'

    peftConfig = LoraConfig(
        r=args.loraR,
        lora_alpha=args.loraAlpha,
        lora_dropout=args.loraDropout,
        target_modules=['q_proj', 'v_proj'],
        bias='none',
        task_type='CAUSAL_LM',
    )

    trainset = load_multiple_datasets(trainPathList, shuffle=True)
    #trainsets = splitDataset(trainset, args.trainNum, accelerator.num_processes)

    #trainset = trainsets[accelerator.process_index]

    dataCollator = DataCollatorForCompletionOnlyLM('[/INST]', tokenizer=tokenizer, mlm=False)

    trainingArgs = MyTrainingArguments(
        output_dir=outputPath,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        #per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=500,
        save_total_limit=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_dir=logPath,
        logging_steps=1,
        report_to="wandb",
        #gradient_checkpointing=True,
        run_name=args.model,
        bf16=True,
        #model_max_length=args.maxTokenLength,
        dataset_text_field='text',
        #max_seq_length=args.maxTokenLength,
        max_seq_length=args.maxTokenLength + 10,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        #max_seq_length=args.maxTokenLength,
        peft_config=peftConfig,
        processing_class=tokenizer,
        #dataset_text_field='text',
        data_collator=dataCollator,
        args=trainingArgs,
    )

    checkpoint = get_last_checkpoint(outputPath)

    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.model.save_pretrained(outputPath)
    tokenizer.save_pretrained(outputPath)
    
    del model
    torch.cuda.empty_cache()
    model = AutoPeftModelForCausalLM.from_pretrained(
        outputPath, torch_dtype=torch.bfloat16
    )

    model = model.merge_and_unload()
    model.eval()
    model.save_pretrained(outputPath)









if __name__ == "__main__":
    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    #torch.distributed.init_process_group(backend='nccl')
    main(args)
    #torch.distributed.destroy_process_group()
