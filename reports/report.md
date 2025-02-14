# Reasoning on Graph report

## 1. Implementation
- evaluate.py, lr.py, gnn.py는 원본 코드를 가져왔다.
- LoRA는 구현하지 않았다.
- 모든 과정에서 모델을 llama-2-7b-chat-hf를 사용하였다.
- 논문에서는 sceneGraphs의 retrieval에서 eCost를 1로 설정하였지만, 원본 코드상에서는 0.5로 설정되어 있었다. 논문을 따라 구현하였다.
- pcst를 돌리고 subGraph를 생성하는 과정에서, 원본 코드는 중복된 node들은 지워주는데, 중복된 edge들은 지워주지 않았다. 의도된 건지는 모르겠으나, 중복된 edge들까지 지워서 구현하였다. 
- 원본 코드에서는 inference-only, prompt-tuning, g-retriever 모두 retrieve를 한 그래프를 사용하는데, 논문에서는 g-retriever에서만 retrieve한 그래프를 사용하므로, 논문을 따라 구현했다.
- 그래프를 textualize할 때, 구분자로 '\n' 대신 '\n\n'을 사용하면 accuracy가 상승한다.

## 2. Evaluation

#### llama-2-7b-chat-hf
- 추론 속도 향상을 위해 max_new_tokens를 512에서 64로 낮추어 inference를 진행했고, 그렇기에 논문에 비해 약간 낮은 결과를 보인다. 
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
```shell
python preprocess.py --model roG --dataset webQsp
python preprocess.py --model roG --dataset cwq
python train.py --dataset sceneGraphs --model graphLlm --useGR
python train.py --dataset webQsp --model graphLlm --useGR
```

- 평가 방식을 같게 하기 위하여 evaluate.py는 내려받아 사용하였다.

| dataset | WebQsp | WebQsp | CWQ | CWQ |
|---------|--------|--------|-----|-----|
| model \ metric| Hit@1 | F1 | Hit@1 | F1 |
| llama-2-7b-chat-hf | 53.87 | 24.59 | 28.07 | 13.42 |
| pretrained roG | 86.36 | 69.32 | 61.00 | 53.89 |
| gRetriever | 0.9350 | 0.4965 | 61.87 |




## 3. Limitation

- 시간관계상 LoRA는 테스트해보지 못하였다.
- 시간관계상 전체 데이터의 일부만 training을 진행하여 accuracy가 낮다. 좀 더 많은 시간을 들이면 accuracy가 높아질 것으로 예상된다.
- 논문의 결과와 비슷한 양상을 보이지만, 완전히 같지는 않기에 수정이 필요하다.
