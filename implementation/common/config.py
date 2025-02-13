import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='llm')
argparser.add_argument('--dataset', type=str, default='webQsp')

# for inference
argparser.add_argument('--testBatchSize', type=int, default=32)
argparser.add_argument('--testNum', type=int, default=0)

# for llm
argparser.add_argument('--modelName', type=str, default='meta-llama/Llama-2-7b-chat-hf')
argparser.add_argument('--maxNewTokens', type=int, default=64)

# for pretrained planning model
argparser.add_argument('--numBeam',  type=int, default=3)
argparser.add_argument('--rogModelName', type=str, default='rmanluo/RoG')