import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='llm')
argparser.add_argument('--dataset', type=str, default='webQsp')

# for inference
argparser.add_argument('--testBatchSize', type=int, default=16)
argparser.add_argument('--testNum', type=int, default=0)

# for llm
argparser.add_argument('--modelName', type=str, default='meta-llama/Llama-2-7b-chat-hf')
argparser.add_argument('--maxNewTokens', type=int, default=64)
argparser.add_argument('--maxTokenLength', type=int, default=512)

# for roG
argparser.add_argument('--numBeam',  type=int, default=3)
argparser.add_argument('--rogModelName', type=str, default='rmanluo/RoG')
argparser.add_argument('--maxPlanTokens', type=int, default=100)

# for train
argparser.add_argument('--trainNum', type=int, default=0)
argparser.add_argument('--loraR', type=int, default=8)
argparser.add_argument('--loraAlpha', type=float, default=16)
argparser.add_argument('--loraDropout', type=float, default=0.05)
argparser.add_argument('--rogModelPath', type=str, default='../output/myRoG/trained_model2')
