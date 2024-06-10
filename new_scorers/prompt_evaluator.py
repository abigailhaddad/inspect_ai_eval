from langchain.prompts import PromptTemplate
from new_scorers.code_from_inspect_ai import InspectChatModel
from inspect_ai import eval, Task, task
from inspect_ai.model import get_model
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
import asyncio
from inspect_ai.dataset import Sample
from typing import Dict, Tuple
from ast import literal_eval
from langchain_core.messages import HumanMessage
import os

class PromptEvaluator:
    """
    A class to evaluate prompts using an AI model.
    """

    def __init__(self, model):
        """
        Initialize the PromptEvaluator with the provided model.
        
        Args:
            model: The AI model used for evaluating prompts.
        """
        self.model = model

    @staticmethod
    def _parse_prompt():
        """
        Generate the prompt template for evaluating the input and target text.

        Returns:
            PromptTemplate: The prompt template.
        """
        return PromptTemplate(
            input_variables=["target_text", "input_text"],
            template="{target_text}\n\nInput: {input_text}\nOutput:"
        )

    async def __call__(self, input_text, target_text):
        """
        Evaluate the input and target text asynchronously and return the result.

        Args:
            input_text (str): The input text.
            target_text (str): The target text.

        Returns:
            int: The evaluation result.
        """
        prompt = self._parse_prompt().format(target_text=target_text, input_text=input_text)
        final_result = (await self.model._agenerate([HumanMessage(content=prompt)])).generations[0].text.strip()
        return self.process_data(final_result)

    def process_data(self, final_result):
        """
        Process the final result and determine the pass value.

        Args:
            final_result (str): The final result from the model.

        Returns:
            int: The pass value (1 if "PASS" is in the result, otherwise 0).
        """
        pass_value = 1 if "PASS" in final_result else 0
        return pass_value

class PromptEvaluatorWrapper(Scorer):
    """
    A wrapper class for the PromptEvaluator to integrate with the scoring system.
    """

    def __init__(self, model):
        """
        Initialize the PromptEvaluatorWrapper with the provided model.
        
        Args:
            model: The AI model used for evaluating prompts.
        """
        self.model = InspectChatModel()
        self.prompt_scorer = PromptEvaluator(self.model)

    async def __call__(self, state: TaskState, target: Sample):
        """
        Process the state and target to calculate the score.

        Args:
            state (TaskState): The current task state.
            target (Sample): The target sample.

        Returns:
            Score: The calculated score.
        """
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
    """
    Metric function to calculate the pass rate.

    Returns:
        function: The metric function.
    """
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
    """
    Create a scorer for the prompt evaluator.

    Args:
        model: The AI model used for evaluating prompts.

    Returns:
        Scorer: The prompt evaluator scorer.
    """
    return PromptEvaluatorWrapper(model)

@task
def prompt_evaluator_eval():
    """
    Create a classification evaluation task.

    Returns:
        Task: The classification evaluation task.
    """
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
