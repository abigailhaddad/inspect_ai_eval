import sys


import random
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
os.environ['PYTHONIOENCODING'] = 'utf-8'


class FactComparator:
    def __init__(self, model):
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=ComparisonResult)

    async def __call__(self, context, answer):
        return await self.process_data(context, answer)

    async def process_data(self, context, answer):
        context_list = (await self.model._agenerate([HumanMessage(content=self._parse_prompt().format(text=context))])).generations[0].text
        answer_list = (await self.model._agenerate([HumanMessage(content=self._parse_prompt().format(text=answer))])).generations[0].text
        print(type(self.model))
        comparison_result = self.parser.parse((await self.model._agenerate([HumanMessage(content=self._compare_prompt().format(context_list=context_list, answer_list=answer_list))])).generations[0].text)

        return {
            "context_list": context_list,
            "answer_list": answer_list,
            "comparison_result": comparison_result,
        }

    def calculate_metrics(self, comparison_result):
        facts_in_both_count = len(comparison_result.facts_in_both)
        facts_only_in_answer_count = len(comparison_result.facts_only_in_answer)
        facts_only_in_context_count = len(comparison_result.facts_only_in_context)

        total_answer_facts = facts_in_both_count + facts_only_in_answer_count
        total_context_facts = facts_in_both_count + facts_only_in_context_count

        groundedness = facts_in_both_count / total_answer_facts * 100 if total_answer_facts > 0 else 0
        thoroughness = facts_in_both_count / total_context_facts * 100 if total_context_facts > 0 else 0

        return {
            "groundedness": groundedness,
            "thoroughness": thoroughness,
        }
    @staticmethod
    def _parse_prompt():
        return PromptTemplate(
            input_variables=["text"],
            template="""
            Here is a text that may contain one or more facts:

            <text>
            {text}
            </text>

            Please parse this text into a list of individual facts. If a sentence contains multiple facts, break it up into separate sentences as needed so that each sentence contains only one fact.

            If any of the facts contain pronouns and the pronoun reference is clear, replace the pronoun with the noun it refers to. If the pronoun reference is ambiguous, leave the pronoun as is.

        Return the final list of parsed and pronoun-replaced facts inside <facts> tags, with each fact on its own line. Do not include any additional commentary or explanation, including about pronoun changes, number of facts, or truth value of the facts.
        """,
        )

    @staticmethod
    def _compare_prompt():
        return PromptTemplate(
            input_variables=["context_list", "answer_list"],
            template="""
            You will be comparing facts between a context and an answer to determine which facts are shared and which are unique to each.

            Here is the context:

            <context>
            {context_list}
            </context>

            And here is the answer: 

            <answer>
            {answer_list}
            </answer>

            Carefully analyze the facts presented in the context and answer, focusing on the semantic meaning rather than the exact wording.

            Then, output a dictionary with the following keys and corresponding lists of facts as values:

            1. "facts_in_both": A list of facts that are present in both the context and the answer

            2. "facts_only_in_answer": A list of facts that are only present in the answer 

            3. "facts_only_in_context": A list of facts that are only present in the context

            Remember, the facts do not need to be worded identically to be considered the same. Focus on whether the core meaning is shared or unique.  A fact in the context may be expressed in different terms in the answer, or multiple facts in one may combine to express a single fact in the other.

            Provide your results in this format:

            {{
                "facts_in_both": [
                    "Fact 1 present in both",
                    "Fact 2 present in both"
                ],
                "facts_only_in_answer": [
                    "Fact 1 only in answer",
                    "Fact 2 only in answer"  
                ],
                "facts_only_in_context": [
                    "Fact 1 only in context",
                    "Fact 2 only in context"
                ]
            }}
            """,
        )


class ComparisonResult(BaseModel):
    facts_in_both: list[str] = Field(default_factory=list, description="List of facts present in both context and answer")
    facts_only_in_answer: list[str] = Field(default_factory=list, description="List of facts only present in the answer")
    facts_only_in_context: list[str] = Field(default_factory=list, description="List of facts only present in the context")

class ModelComparator:
    def __init__(self, model):
        self.inspect_model = InspectChatModel()
        self.comparator = FactComparator(self.inspect_model)

    async def run_and_compare(self, target_statement, input_statement):
        try:
            result = await self.comparator(target_statement, input_statement)
            metrics = self.comparator.calculate_metrics(result["comparison_result"])
            groundedness_model = metrics['groundedness']
            thoroughness_model = metrics['thoroughness']
            context_list = result["context_list"]
            answer_list = result["answer_list"]
            comparison_result = result["comparison_result"]
            model_error = None
        except Exception as e:
            groundedness_model = None
            thoroughness_model = None
            context_list = None
            answer_list = None
            comparison_result = None
            model_error = str(e)

        return {
            'Groundedness (Model)': groundedness_model,
            'Thoroughness (Model)': thoroughness_model,
            'Context List': context_list,
            'Answer List': answer_list,
            'Facts in Both': comparison_result.facts_in_both if comparison_result else None,
            'Facts Only in Answer': comparison_result.facts_only_in_answer if comparison_result else None,
            'Facts Only in Context': comparison_result.facts_only_in_context if comparison_result else None,
            'Model Error': model_error
        }


class FactComparatorScorer:
    def __init__(self, model):
        self.model = model
        self.fact_comparator = FactComparator(model)

    async def __call__(self, state: TaskState, target: Sample):
        try: 
            context = state.output.choices[0].message.content
        except: 
            context = state.input
        target_text = target.target

        result = await self.fact_comparator.process_data(context, target_text)
        metrics = self.fact_comparator.calculate_metrics(result["comparison_result"])

        scorer_value = {
            "groundedness": metrics["groundedness"],
            "thoroughness": metrics["thoroughness"],
        }

        explanation = str(result) + f"\nModel Output: {context}"

        return Score(
            value=scorer_value,
            explanation=explanation,
        )
        
@metric
def thoroughness():
  def metric(scores: list[Score]) -> float:
    total = 0.0
    for item in scores:
      metadata = item.metadata
      if metadata is not None:
          total += float(metadata["thoroughness"])
    return total / float(len(scores))
  return metric

@metric
def groundedness():
  def metric(scores: list[Score]) -> float:
    total = 0.0
    for item in scores:
        metadata = item.metadata
        if metadata is not None:
            total += float(metadata["groundedness"])
    return total / float(len(scores))
  return metric

    
@scorer(metrics=[groundedness(), thoroughness()])
def fact_comparator_scorer(model) -> Scorer:
  
  async def score(state: TaskState, target: Target) -> Score:

    # Create an instance of the scorer
    model = InspectChatModel()
    fact_comparator_scorer = FactComparatorScorer(model)

    # Call the scorer
    score = await fact_comparator_scorer(state, target)
    print(score)

    # Ignore the actual processing and return a dummy value
    grounded_score = score.value['groundedness']
    thorough_score = score.value['thoroughness']
    explanation = score.explanation

    answer = state.output.completion

    return Score(
        value=f"G:{grounded_score} : T:{thorough_score}", # make a better string?
        answer=answer,
        explanation= "nothing",
        metadata = {
           "thoroughness": thorough_score,
           "groundedness": grounded_score,
            "stuff": explanation
        }
    )

  return score

def sample_sample():
    samples = [
    Sample(
        input="How old is the sun?",
        target="The sun is approximately 4.6 billion years old. It's a mid-sized star.",
        description="Very basic question.",
        id="case1"
    )]
    return(samples)

@task
def my_eval():
    samples = sample_sample()
    SYSTEM_MESSAGE = "Please answer the question being asked."
    return Task(
        dataset=samples,
        plan=[
            system_message(SYSTEM_MESSAGE),
            generate()
        ],
        scorer=fact_comparator_scorer(model=get_model()),
    )


def compare_metrics(cases: Dict[str, Dict[str, Tuple[str, str, Dict[str, float], str]]]):
    data = []
    model_comparator = ModelComparator(model='openai/gpt-4')

    # Create an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for case_name, case_data in cases.items():
            input_statement = case_data['input']
            target_statement = case_data['target']
            true_metrics = case_data['true_metrics']
            description = case_data['description']

            # Run the coroutine on the loop
            model_results = loop.run_until_complete(model_comparator.run_and_compare(target_statement, input_statement))

            groundedness_true = true_metrics['groundedness']
            thoroughness_true = true_metrics['thoroughness']

            data.append({
                'Case': case_name,
                'Input Statement': input_statement,
                'Target Statement': target_statement,
                'Description': description,
                'Groundedness (True)': groundedness_true,
                'Thoroughness (True)': thoroughness_true,
                **model_results
            })
    finally:
        # Close the loop after all iterations are complete
        loop.close()

    df = pd.DataFrame(data)
    return df



from langchain.prompts import PromptTemplate
from code_from_inspect_ai import InspectChatModel

class PromptScorer:
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

class PromptScorerWrapper(Scorer):
    def __init__(self, model):
        self.model = InspectChatModel()
        self.prompt_scorer = PromptScorer(self.model)

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
    return PromptScorerWrapper(model)

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
    print("here")
    return Task(
        dataset=samples,
        plan=[
            system_message(SYSTEM_MESSAGE),
            generate()
        ],
        scorer=prompt_scorer(model=get_model()),
    )