import unittest
import asyncio
from prompt_evaluator import PromptEvaluator
from code_from_inspect_ai import InspectChatModel
import os

os.environ['INSPECT_EVAL_MODEL'] = 'openai/gpt-4'
os.environ['INSPECT_MODEL_NAME'] = 'openai/gpt-4'

class TestPromptEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = PromptEvaluator(InspectChatModel())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_evaluate(self):
        test_cases = [
            {
                'input_text': 'The Earth is 3 billion years old.',
                'target_text': 'Return PASS if the answer contains that the sun is 4.6 billion years old, return FAIL otherwise.',
                'expected_score': 0
            },
            {
                'input_text': 'The Sun is approximately 4.6 billion years old and is a mid-sized star.',
                'target_text': 'Return PASS if the answer contains that the sun is 4.6 billion years old, return FAIL otherwise.',
                'expected_score': 1
            },
            # Add more test cases here
        ]

        for case in test_cases:
            input_text = case['input_text']
            target_text = case['target_text']
            expected_score = case['expected_score']

            actual_score = self.loop.run_until_complete(self.evaluator(input_text, target_text))
            self.assertEqual(actual_score, expected_score)

if __name__ == '__main__':
    unittest.main()