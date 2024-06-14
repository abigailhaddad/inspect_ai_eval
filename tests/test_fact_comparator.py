import unittest
import os
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import system_message, generate
from inspect_ai.model import get_model
from inspect_ai.scorer import Scorer
from inspect_ai.log._log import EvalMetric
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local application imports
from inspect_ai_scorers.fact_comparator import fact_comparator_scorer
from inspect_ai_scorers.code_from_inspect_ai import InspectChatModel

class TestFactComparatorEvaluation(unittest.TestCase):
    def setUp(self):
        self.model = 'openai/gpt-4'
        os.environ['INSPECT_EVAL_MODEL'] = self.model
        os.environ['INSPECT_MODEL_NAME'] = self.model
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        self.inspect_model = InspectChatModel()

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
            
            # Run the evaluation
            eval_results = eval(task, model=self.model)
            
            # Check if eval_results is a list
            self.assertIsInstance(eval_results, list)
            
            # Check if the first item in the results list is an EvalLog
            result = eval_results[0]
            self.assertTrue(hasattr(result, 'eval'))
            self.assertTrue(hasattr(result, 'plan'))
            self.assertTrue(hasattr(result, 'results'))
            self.assertTrue(hasattr(result, 'stats'))
            self.assertTrue(hasattr(result, 'samples'))

            # Check the structure of 'results' field
            results = result.results
            self.assertTrue(hasattr(results, 'scorer'))
            self.assertTrue(hasattr(results, 'metrics'))
            self.assertIsInstance(results.metrics, dict)

            # Check the 'metrics' field
            metrics = results.metrics
            self.assertIn('groundedness', metrics)
            self.assertIn('thoroughness', metrics)
            self.assertIsInstance(metrics['groundedness'], EvalMetric)
            self.assertIsInstance(metrics['thoroughness'], EvalMetric)

            # Check the 'samples' field
            samples = result.samples
            self.assertIsInstance(samples, list)
            sample = samples[0]
            self.assertTrue(hasattr(sample, 'id'))
            self.assertTrue(hasattr(sample, 'input'))
            self.assertTrue(hasattr(sample, 'target'))
            self.assertTrue(hasattr(sample, 'messages'))
            self.assertTrue(hasattr(sample, 'output'))
            self.assertTrue(hasattr(sample, 'score'))

            # Check the 'score' field in the sample
            score = sample.score
            self.assertTrue(hasattr(score, 'value'))
            self.assertTrue(hasattr(score, 'answer'))
            self.assertTrue(hasattr(score, 'explanation'))
            self.assertTrue(hasattr(score, 'metadata'))

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
        scorer=fact_comparator_scorer(),
    )

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
