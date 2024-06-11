import asyncio
import os

os.environ['INSPECT_EVAL_MODEL'] = 'openai/gpt-4'
os.environ['INSPECT_MODEL_NAME'] = 'openai/gpt-4'

from new_scorers.prompt_evaluator import PromptEvaluator

import unittest
import os
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log._log import EvalLog, EvalResults, EvalSample
from inspect_ai.scorer._metric import Score
from inspect_ai._eval.eval import EvalLogs
from inspect_ai.solver import generate, system_message
from inspect_ai.model import get_model

from new_scorers.prompt_evaluator import prompt_scorer

model = InspectChatModel()
evaluator = PromptEvaluator(model)

test_cases = [
    {
        'input_text': 'The Earth is 3 billion years old.',
        'target_text': 'Return PASS if the answer contains that the sun is 4.6 billion years old, return FAIL otherwise.',
        'expected_score': 0
    },
    {
        'input_text': 'The Sun is approximately 4.6 billion years old and is a mid-sized star.',
        'target_text': 'Return PASS if the answer mentions that the sun is 4.6 billion years old, return FAIL otherwise.',
        'expected_score': 1
    },
    # Add the rest of your test cases here
]

async def evaluate_example(example):
    input_text = example['input_text']
    target_text = example['target_text']
    expected_score = example['expected_score']

    try:
        actual_score = await evaluator(input_text, target_text)
        model_result = "PASS" if actual_score == 1 else "FAIL"
        test_passed = actual_score == expected_score
        return test_passed, model_result, expected_score
    except Exception as e:
        return False, str(e), expected_score

def main():
    print("Running all examples...\n")
    results = asyncio.run(run_all_examples())

    print("\nPerformance Report:")
    for i, (example, (test_passed, model_result, expected)) in enumerate(zip(test_cases, results), 1):
        print(f"Example {i}:")
        print(f"  Input: '{example['input_text']}'")
        print(f"  Target: '{example['target_text']}'")
        print(f"  Model Result: {model_result}")
        if test_passed:
            print("  Test Result: PASS")
        else:
            print(f"  Test Result: FAIL (Expected: {'PASS' if expected == 1 else 'FAIL'})")
        print("")

async def run_all_examples():
    tasks = [evaluate_example(example) for example in test_cases]
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    main()
