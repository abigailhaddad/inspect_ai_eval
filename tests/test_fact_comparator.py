import unittest
import asyncio
from new_scorers.fact_comparator import FactComparator, ModelComparator
from new_scorers.code_from_inspect_ai import InspectChatModel
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

    def test_case2(self):
        """
        Test case 2: Basic use case with mild rephrasing.
        """
        case_data = {
            'context_text': 'The Sun, a medium-sized star, is located at the center of our Solar System and is approximately 4.6 billion years old.',
            'answer_text': 'The sun is a mid-sized star which has existed for about 4.6 billion years.',
            'true_metrics': {'groundedness': 67, 'thoroughness': 100},
            'description': 'This is a basic use case with mild rephrasing.'
        }
        self.run_test_case(case_data)

    def test_case3(self):
        """
        Test case 3: Simple restructuring and clarification.
        """
        case_data = {
            'context_text': 'Sally is Rachel\'s cat.',
            'answer_text': 'Sally is a cat. Rachel is her owner.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This case involves simple restructuring and clarification.'
        }
        self.run_test_case(case_data)

    def test_case4(self):
        """
        Test case 4: Change in comparative perspective.
        """
        case_data = {
            'context_text': 'Sally is larger than Stan.',
            'answer_text': 'Stan is smaller than Sally.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This case demonstrates a change in comparative perspective.'
        }
        self.run_test_case(case_data)

    def test_case5(self):
        """
        Test case 5: Unit conversion and synonym use.
        """
        case_data = {
            'context_text': 'The average temperature today is 20 degrees Celsius.',
            'answer_text': 'The mean temperature today is 68 degrees Fahrenheit.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This case involves unit conversion and synonym use.'
        }
        self.run_test_case(case_data)

    def test_case6(self):
        """
        Test case 6: Unit conversion and synonym use with incorrect facts.
        """
        case_data = {
            'context_text': 'The average temperature today is 20 degrees Celsius.',
            'answer_text': 'The average temperature today is 50 degrees Celsius.',
            'true_metrics': {'groundedness': 0, 'thoroughness': 0},
            'description': 'This case involves unit conversion and synonym use.'
        }
        self.run_test_case(case_data)

    def test_case7(self):
        """
        Test case 7: Dual meaning of "sanctioned."
        """
        case_data = {
            'context_text': 'The company has an ATO now, so they have been sanctioned by the government and you can work with them.',
            'answer_text': 'The company has been sanctioned by the government in response to recent lawbreaking activity.',
            'true_metrics': {'groundedness': 0, 'thoroughness': 0},
            'description': 'This case uses "sanctioned" in a way that highlights its dual meaning: approved or penalized.'
        }
        self.run_test_case(case_data)

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

if __name__ == '__main__':
    unittest.main()
