# test_fact_comparator.py
import unittest
import asyncio
from fact_comparator import FactComparator, ModelComparator
from code_from_inspect_ai import InspectChatModel
import os

os.environ['INSPECT_EVAL_MODEL'] = 'openai/gpt-4'
os.environ['INSPECT_MODEL_NAME'] = 'openai/gpt-4'

class TestFactComparatorMetrics(unittest.TestCase):
    def setUp(self):
        self.model = InspectChatModel()
        self.fact_comparator = FactComparator(self.model)
        self.model_comparator = ModelComparator(self.model)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_case1(self):
        case_data = {
            'input': 'The Sun is a medium-sized star. It\'s about 4.6 billion years old.',
            'target': 'The sun is approximately 4.6 billion years old. It\'s a mid-sized star.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This is a basic use case with pronouns and mild rephrasing.'
        }
        self.run_test_case(case_data)

    def test_case2(self):
        case_data = {
            'input': 'The Sun, a medium-sized star, is located at the center of our Solar System and is approximately 4.6 billion years old.',
            'target': 'The sun is a mid-sized star which has existed for about 4.6 billion years.',
            'true_metrics': {'groundedness': 67, 'thoroughness': 100},
            'description': 'This is a basic use case with mild rephrasing.'
        }
        self.run_test_case(case_data)

    def test_case3(self):
        case_data = {
            'input': 'Sally is Rachel\'s cat.',
            'target': 'Sally is a cat. Rachel is her owner.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This case involves simple restructuring and clarification.'
        }
        self.run_test_case(case_data)

    def test_case4(self):
        case_data = {
            'input': 'Sally is larger than Stan.',
            'target': 'Stan is smaller than Sally.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This case demonstrates a change in comparative perspective.'
        }
        self.run_test_case(case_data)

    def test_case5(self):
        case_data = {
            'input': 'the average temperature today is 20 degrees celsius.',
            'target': 'the mean temperature today is 68 degrees fahrenheit.',
            'true_metrics': {'groundedness': 100, 'thoroughness': 100},
            'description': 'This case involves unit conversion and synonym use.'
        }
        self.run_test_case(case_data)

    def test_case6(self):
        case_data = {
            'input': 'the average temperature today is 20 degrees celsius.',
            'target': 'the average temperature today is 50 degrees celsius.',
            'true_metrics': {'groundedness': 0, 'thoroughness': 0},
            'description': 'This case involves unit conversion and synonym use.'
        }
        self.run_test_case(case_data)

    def test_case7(self):
        case_data = {
            'input': 'The company has an ATO now, so they have been sanctioned by the government and you can work with them.',
            'target': 'The company has been sanctioned by the government in response to recent lawbreaking activity.',
            'true_metrics': {'groundedness': 0, 'thoroughness': 0},
            'description': 'This case uses "sanctioned" in a way that highlights its dual meaning: approved or penalized.'
        }
        self.run_test_case(case_data)

    def run_test_case(self, case_data):
        input_statement = case_data['input']
        target_statement = case_data['target']
        true_metrics = case_data['true_metrics']

        model_results = self.loop.run_until_complete(self.model_comparator.run_and_compare(target_statement, input_statement))
        groundedness_model = round(model_results['Groundedness (Model)'])
        thoroughness_model = round(model_results['Thoroughness (Model)'])

        self.assertEqual(groundedness_model, true_metrics['groundedness'])
        self.assertEqual(thoroughness_model, true_metrics['thoroughness'])

if __name__ == '__main__':
    unittest.main()