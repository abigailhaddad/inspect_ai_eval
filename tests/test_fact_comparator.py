import unittest
import asyncio
from new_scorers.fact_comparator import FactComparator, ModelComparator
import sys
from new_scorers.code_from_inspect_ai import InspectChatModel
import random
from inspect_ai.dataset import Sample
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
from typing import Dict, Tuple
from ast import literal_eval
from langchain_core.messages import HumanMessage
import os
from new_scorers.fact_comparator import fact_comparator_scorer
import os

os.environ['INSPECT_EVAL_MODEL'] = 'openai/gpt-4'
os.environ['INSPECT_MODEL_NAME'] = 'openai/gpt-4'

class TestFactComparatorMetrics(unittest.TestCase):
    """
    A class to test the FactComparator metrics.
    """

    def setUp(self):
        """
        Set up the test environment by initializing the model, fact comparator, and model comparator.
        """
        self.model = InspectChatModel()
        self.fact_comparator = FactComparator(self.model)
        self.model_comparator = ModelComparator(self.model)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """
        Tear down the test environment by closing the event loop.
        """
        self.loop.close()

    def test_case1(self):
        """
        Test case 1: Basic use case with pronouns and mild rephrasing.
        """
        case_data = {
            'context_text': 'The Sun is a medium-sized star. It\'s about 4.6 billion years old.',
            'answer_text': 'The sun is approximately 4.6 billion years old. It\'s a mid-sized star.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This is a basic use case with pronouns and mild rephrasing.'
        }
        self.run_test_case(case_data)

    # Add other test cases here with similar structure

    def run_test_case(self, case_data):
        """
        Run a test case and assert the metrics.

        Args:
            case_data (dict): The case data containing context, answer, true metrics, and description.
        """
        context_text = case_data['context_text']
        answer_text = case_data['answer_text']
        true_metrics = case_data['true_metrics']

        model_results = self.loop.run_until_complete(self.model_comparator.run_and_compare(context_text, answer_text))
        groundedness_model = round(model_results['Groundedness (Model)'])
        thoroughness_model = round(model_results['Thoroughness (Model)'])

        self.assertEqual(groundedness_model, true_metrics['groundedness'])
        self.assertEqual(thoroughness_model, true_metrics['thoroughness'])

class TestFactComparatorEvaluation(unittest.TestCase):
    def setUp(self):
        self.model = InspectChatModel()

    def test_create_sample_data(self):
        samples = sample_data()
        self.assertIsInstance(samples, list)
        self.assertIsInstance(samples[0], Sample)

    def test_fact_comparator_eval_task(self):
        try:
            task = fact_comparator_eval()
            self.assertIsInstance(task, Task)

            self.assertTrue(hasattr(task, 'scorer'))
            self.assertIsInstance(task.scorer, Scorer)
            
            # Print the scorer details for debugging
            print(type(task))
            
            # Run the evaluation
            eval_results = eval(task, model="openai/gpt-4")
            print(f"Eval results: {eval_results}")

        except Exception as e:
            self.fail(f"fact_comparator_eval test failed: {e}")

def sample_data():
    """
    Create sample data for evaluation.

    Returns:
        list: A list of sample data.
    """
    samples = [
        Sample(
            input="How old is the sun?",
            target="The sun is approximately 4.6 billion years old. It's a mid-sized star.",
            description="Very basic question.",
            id="case1"
        )
    ]
    return samples

@task
def fact_comparator_eval():
    """
    Create an evaluation task for the fact comparator.

    Returns:
        Task: The evaluation task.
    """
    samples = sample_data()
    SYSTEM_MESSAGE = "Please answer the question being asked."
    return Task(
        dataset=samples,
        plan=[
            system_message(SYSTEM_MESSAGE),
            generate()
        ],
        scorer=fact_comparator_scorer(model=get_model()),
    )

if __name__ == '__main__':
    unittest.main()