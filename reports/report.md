# Reasoning on Graph report

## 1. Implementation
- evaluate.py는 원본 코드를 가져왔다.
- 배치 단위로 돌리기 위해 padding을 이용하면 float16의 경우 nan이 발생하는 오류가 나기에 bfloat16으로 구현했다.
- explain 버전은 따로 학습시키지 않았다.

## 2. Evaluation

#### llama-2-7b-chat-hf
- 추론 속도 향상을 위해 max_new_tokens를 512에서 256으로 낮추어 inference를 진행했고, 그렇기에 논문에 비해 약간 낮은 결과를 보인다. 
```shell
accelerate launch --num_processes 2 --gpu_ids 6,7 inference.py --dataset webQsp --maxNewTokens 256
accelerate launch --num_processes 2 --gpu_ids 6,7 inference.py --dataset cwq --maxNewTokens 256
```

#### pretrained roG
```shell
accelerate launch --num_processes 2 --gpu_ids 6,7 inference.py --model roG --dataset webQsp
accelerate launch --num_processes 2 --gpu_ids 6,7 inference.py --model roG --dataset cwq
```

#### roG
- 학습 속도 향상을 위해 max_seq_length를 2048에서 512로 낮추어 training을 진행했다.
```shell
# preprocessing
python preprocess.py --model myRoG --dataset webQsp
python preprocess.py --model myRoG --dataset cwq

# training 
python train.py --model myRoG

# inferencing
accelerate launch --num_processes 2 --gpu_ids 6,7 inference.py --model myRoG --dataset webQsp
accelerate launch --num_processes 2 --gpu_ids 6,7 inference.py --model myRoG --dataset cwq
```

- 평가 방식을 같게 하기 위하여 evaluate.py는 내려받아 사용하였다.

| dataset | WebQsp | WebQsp | CWQ | CWQ |
|---------|--------|--------|-----|-----|
| model \ metric| Hit@1 | F1 | Hit@1 | F1 |
| llama-2-7b-chat-hf | 58.91 | 23.09 | 33.70 | 13.42 |
| pretrained roG | 86.36 | 69.32 | 61.00 | 53.89 |
| roG | 0.9350 | 0.4965 | 61.87 | 




## 3. Limitation

- webQsp, cwq 데이터셋을 학습하여 해당 데이터셋에 특화된 모델이다.
- 다른 데이터셋으로 테스트를 하면 낮은 성능을 보일 것으로 예상되며, 높은 성능을 위해서는 추가적인 학습 시간이 필요하다.
- 학습시간이 매우 오래걸린다. 
