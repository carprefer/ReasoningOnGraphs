# G-Retriever Design

[Language] Python 3.9

[Os] Linux server(dslab)

[GPU] Tesla P40

[CUDA SDK] 12.4

[PyTorch] 2.5.1

[LLM] LLaMA 2-7b-chat-hf

## Milestone
1. ~~환경 설정~~
2. ~~dataset 모듈 구현~~ 
3. ~~LLM만 사용하여 평가 진행 및 비교~~
4. ~~planning 모듈 구현 및 테스트(w/ pre_trained model)~~
5. ~~retrieve 모듈 구현 및 테스트~~ 
6. ~~reasoning 모듈 구현 및 테스트(w/ pre_trained model)~~
7. ~~training 데이터 preprocessing~~
8. planning 학습
9. reasoning 학습
10. 전체 평가

## Environment Setup

#### miniconda 가상환경 설정
```shell
conda create --name roG python=3.9 -y
conda activate roG

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers
pip install datasets
pip install accelerate
pip install sentencepiece
pip install protobuf
pip install peft
pip install trl
pip install wandb
```

#### accelerate 설정
```shell
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"

# use accelerator instead of python to execute program
accelerator launch --num_processes 2 --gpu_ids 6,7 inference.py
```

## LLM 설정
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', padding_side='left', use_fast=False)

self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype=torch.bfloat16)
```


