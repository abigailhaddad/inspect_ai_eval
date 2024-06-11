import unittest
import os
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log._log import EvalLog, EvalResults, EvalSample
from inspect_ai.scorer._metric import Score
from inspect_ai._eval.eval import EvalLogs
from inspect_ai.solver import generate, system_message
from inspect_ai.model import get_model
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_scorers.prompt_evaluator import prompt_scorer

@task
def prompt_evaluator_eval():
    """
    Create a classification evaluation task.

    This function creates a task for evaluating a language model's ability to answer a specific question.
    The task includes a single sample with an input question and a target condition for determining whether
    the model's answer is correct.

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

class TestPromptEvaluator(unittest.TestCase):
    """
    Test case for the PromptEvaluator class.

    This class contains tests for the `prompt_evaluator_eval` function, which creates a classification evaluation task
    for the PromptEvaluator class.
    """

    def setUp(self):
        self.model = 'openai/gpt-4'
        os.environ['INSPECT_EVAL_MODEL'] = self.model
        os.environ['INSPECT_MODEL_NAME'] = self.model
        os.environ['PYTHONIOENCODING'] = 'utf-8'

    def test_prompt_evaluator_eval_task(self):
        """
        Test the `prompt_evaluator_eval` function.

        This test method checks if the `prompt_evaluator_eval` function creates a valid `Task` object and verifies
        the structure of the evaluation results returned by running the task.
        """
        try:
            task = prompt_evaluator_eval()
            self.assertIsInstance(task, Task)

            # Run the evaluation
            eval_results = eval(task, model=self.model)
            self.assertIsInstance(eval_results, EvalLogs)

            # Check the first item in the results list
            result = eval_results[0]
            self.assertIsInstance(result, EvalLog)

            # Check the structure of 'results' field
            results = result.results
            self.assertIsInstance(results, EvalResults)

            # Check the 'metrics' field
            metrics = results.metrics
            self.assertIsInstance(metrics, dict)

            # Check the 'samples' field
            samples = result.samples
            self.assertIsInstance(samples, list)

            # Check the first sample
            sample = samples[0]
            self.assertIsInstance(sample, EvalSample)

            # Check the 'score' field in the sample
            score = sample.score
            self.assertIsInstance(score, Score)

        except Exception as e:
            self.fail(f"prompt_evaluator_eval test failed: {e}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
