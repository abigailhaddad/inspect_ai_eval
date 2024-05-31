from langchain.prompts import PromptTemplate
from code_from_inspect_ai import InspectChatModel

from inspect_ai import eval, Task, task
from inspect_ai.model import get_model
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
import asyncio
from inspect_ai.dataset import Sample
from typing import Dict, Tuple
from ast import literal_eval
from langchain_core.messages import HumanMessage
from code_from_inspect_ai import InspectChatModel
from inspect_ai.dataset import Sample
import os

class PromptEvaluator:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _parse_prompt():
        return PromptTemplate(
            input_variables=["target_text", "input_text"],
            template="{target_text}\n\nInput: {input_text}\nOutput:"
        )

    async def __call__(self, input_text, target_text):
        prompt = self._parse_prompt().format(target_text=target_text, input_text=input_text)
        final_result = (await self.model._agenerate([HumanMessage(content=prompt)])).generations[0].text.strip()
        return self.process_data(final_result)

    def process_data(self, final_result):
        pass_value = 1 if "PASS" in final_result else 0
        return pass_value

class PromptEvaluatorWrapper(Scorer):
    def __init__(self, model):
        self.model = InspectChatModel()
        self.prompt_scorer = PromptEvaluator(self.model)

    async def __call__(self, state: TaskState, target: Sample):
        input_text = state.output.choices[0].message.content
        target_text = target.target

        pass_value = await self.prompt_scorer(input_text, target_text)

        return Score(
            value=pass_value,
            answer=state.output.completion,
            metadata={"pass": pass_value}
        )

@metric
def pass_metric():
    def metric(scores: list[Score]) -> float:
        total_pass = 0
        for score in scores:
            metadata = score.metadata
            if metadata is not None:
                total_pass += float(metadata["pass"])
        return total_pass / float(len(scores))
    return metric

@scorer(metrics=[pass_metric()])
def prompt_scorer(model) -> Scorer:
    return PromptEvaluatorWrapper(model)

@task
def classification_eval():
    samples = [
        Sample(
            input="How old is the sun?",
            target="Return PASS if the answer contains that the sun is 4.6 billion years old, return FAIL otherwise.",
            description="Very basic question.",
            id="case1"
        )
    ]
    SYSTEM_MESSAGE = "Please answer the question being asked."
    return Task(
        dataset=samples,
        plan=[
            system_message(SYSTEM_MESSAGE),
            generate()
        ],
        scorer=prompt_scorer(model=get_model()),
    )