import random
from common.utils import *

class Prompter():
    def __init__(self, tokenizer, maxTokenLength):
        self.tokenizer = tokenizer
        self.maxTokenLength = maxTokenLength
        self.generatePrompt = {
            'llm': self.llmPrompt,
            'plan': self.rogPlanningPrompt,
            'roG': self.rogReasoningPrompt,
            'myRoG': self.rogReasoningPrompt,
        }

    
    def isPromptFit(self, prompt):
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(tokens) <= self.maxTokenLength

    def llmPrompt(self, data: dict):
        prompt = """[INST] <<SYS>>\n<</SYS>>\n{instruction}\n\nQuestion:\n{question} [/INST]"""
        instruction = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
        question = data['question']

        if not question.endswith('?'):
            question += '?'

        return prompt.format(instruction=instruction, question=question)

    def rogPlanningPrompt(self, data: dict):
        prompt = """[INST] <<SYS>>\n<</SYS>>\n{instruction}{question} [/INST]"""
        instruction = """Please generate a valid relation path that can be helpful for answering the following question: """

        return prompt.format(instruction=instruction, question=data['question'])

    def rogReasoningPrompt(self, data: dict):
        prompt = """[INST] <<SYS>>\n<</SYS>>\n{instruction}\n\nReasoning Paths:\n{context}\n\nQuestion:\n{question} [/INST]"""
        instruction = """Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
        question = data['question']
        if not question.endswith('?'):
            question += '?'

        paths = [path2string(p) for p in data['reasoningPaths']]
        random.shuffle(paths)
        newPaths = []
        for path in paths:
            tmpContext = '\n'.join(newPaths + [path])
            tmpPrompt = prompt.format(instruction=instruction, context=tmpContext, question=question)
            if not self.isPromptFit(tmpPrompt):
                break
            newPaths.append(path)

        return prompt.format(instruction=instruction, context='\n'.join(newPaths), question=question)

    def generatePrompts(self, datas: list[dict], type:str):
        return [self.generatePrompt[type](d) for d in datas]