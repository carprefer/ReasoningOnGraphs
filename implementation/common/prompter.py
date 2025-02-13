

def llmPrompt(data: dict):
    prompt = """[INST] <<SYS>>\n<</SYS>>\n{instruction}\n\nQuestion:\n{question} [/INST]"""
    instruction = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    question = data['question']

    if not question.endswith('?'):
        question += '?'

    return prompt.format(instruction=instruction, question=question)

def rogPlanningPrompt(data: dict):
    prompt = """[INST] <<SYS>>\n<</SYS>>\n{instruction}{question} [/INST]"""
    instruction = """Please generate a valid relation path that can be helpful for answering the following question: """

    return prompt.format(instruction=instruction, question=data['question'])


generatePrompt = {
    'llm': llmPrompt,
    'rogPlan': rogPlanningPrompt
}

def generatePrompts(datas: list[dict], type:str):
    return [generatePrompt[type](d) for d in datas]