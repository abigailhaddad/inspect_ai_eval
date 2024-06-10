import unittest
import asyncio
from new_scorers.prompt_evaluator import PromptEvaluator
from new_scorers.code_from_inspect_ai import InspectChatModel
import os
import contextlib

os.environ['INSPECT_EVAL_MODEL'] = 'openai/gpt-4'
os.environ['INSPECT_MODEL_NAME'] = 'openai/gpt-4'

class TestPromptEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = PromptEvaluator(InspectChatModel())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.run_until_complete(self.close_async_tasks())
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()

    async def close_async_tasks(self):
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def test_evaluate(self):
        test_cases = [
            {
                'input_text': 'The Earth is 3 billion years old.',
                'target_text': 'Return PASS if the answer mentions that the sun is 4.6 billion years old, return FAIL otherwise.',
                'expected_score': 0
            },
            {
                'input_text': 'The Sun is approximately 4.6 billion years old and is a mid-sized star.',
                'target_text': 'Return PASS if the answer mentions that the sun is 4.6 billion years old, return FAIL otherwise.',
                'expected_score': 1
            },
            {
                'input_text': 'The Moon orbits the Earth every 27.3 days.',
                'target_text': 'Return PASS if the answer mentions the orbital period of the Moon is between 27 and 28 days, return FAIL otherwise.',
                'expected_score': 1
            },
            {
                'input_text': 'Water boils at 90 degrees Celsius at sea level.',
                'target_text': 'Return PASS if the answer mentions that water boils at 100 degrees Celsius at sea level, return FAIL otherwise.',
                'expected_score': 0
            },
            {
                'input_text': 'The capital of France is Paris.',
                'target_text': 'Return PASS if the answer mentions that the capital of France is Paris, return FAIL otherwise.',
                'expected_score': 1
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                input_text = case['input_text']
                target_text = case['target_text']
                expected_score = case['expected_score']

                actual_score = self.loop.run_until_complete(self.evaluator(input_text, target_text))
                self.assertEqual(actual_score, expected_score)

if __name__ == '__main__':
    unittest.main()
