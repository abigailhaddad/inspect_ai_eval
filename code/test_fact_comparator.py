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

    def test_metrics(self):
        cases = {
            'case1': {
                'input': 'The Sun is a medium-sized star. It\'s about 4.6 billion years old.',
                'target': 'The sun is approximately 4.6 billion years old. It\'s a mid-sized star.',
                'true_metrics': {'groundedness': 100, 'thoroughness': 100},
                'description': 'This is a basic use case with pronouns and mild rephrasing.'
            },
            # Add the rest of the cases here
        }

        for case_name, case_data in cases.items():
            input_statement = case_data['input']
            target_statement = case_data['target']
            true_metrics = case_data['true_metrics']
            description = case_data['description']

            model_results = self.loop.run_until_complete(self.model_comparator.run_and_compare(target_statement, input_statement))
            groundedness_model = model_results['Groundedness (Model)']
            thoroughness_model = model_results['Thoroughness (Model)']

            self.assertEqual(groundedness_model, true_metrics['groundedness'])
            self.assertEqual(thoroughness_model, true_metrics['thoroughness'])

if __name__ == '__main__':
    unittest.main()