import argparse
import asyncio
import os
import logging
from datetime import datetime

from new_scorers.prompt_evaluator import PromptEvaluator
from new_scorers.code_from_inspect_ai import InspectChatModel


# Set up argument parsing
parser = argparse.ArgumentParser(description="Run Prompt Evaluator examples.")
parser.add_argument('--model', type=str, default='openai/gpt-4', help='The model name to use for evaluation.')
args = parser.parse_args()

# Set environment variable for model name
os.environ['INSPECT_MODEL_NAME'] = args.model

# Logging configuration
if not os.path.exists('logs'):
    os.makedirs('logs')

log_filename = f"logs/prompt_evaluator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

model = InspectChatModel()
evaluator = PromptEvaluator(model)

test_cases = [
    {
        'description': 'Checking the number of times the letter "r" appears in the word "carryforward". Pass case.',
        'input_text': 'The letter r appears four times in the word carryforward.',
        'target_text': 'Return PASS if the answer says that the letter r is in the word carryforward four times, return FAIL otherwise.',
        'expected_score': 1
    },
    {
        'description': 'Checking the number of times the letter "r" appears in the word "carryforward". Fail case.',
        'input_text': 'The letter r appears three times in the word carryforward.',
        'target_text': 'Return PASS if the answer says that the letter r is in the word carryforward four times, return FAIL otherwise.',
        'expected_score': 0
    },
    {
        'description': 'Checking the time difference between Boston and Copenhagen. Pass case.',
        'input_text': 'The time in Copenhagen is 6 PM.',
        'target_text': 'Return PASS if the answer says that the time in Copenhagen is 6 PM, return FAIL otherwise.',
        'expected_score': 1
    },
    {
        'description': 'Checking the time difference between Boston and Copenhagen. Fail case.',
        'input_text': 'The time in Copenhagen is 7 PM.',
        'target_text': 'Return PASS if the answer says that the time in Copenhagen is 6 PM, return FAIL otherwise.',
        'expected_score': 0
    },
    {
        'description': 'Finding the letter on the keyboard based on the given instructions. Pass case.',
        'input_text': 'The letter is "y".',
        'target_text': 'Return PASS if the answer says that "y" is the letter, return FAIL otherwise.',
        'expected_score': 1
    },
    {
        'description': 'Finding the letter on the keyboard based on the given instructions. Fail case.',
        'input_text': 'The letter is "t".',
        'target_text': 'Return PASS if the answer says that "y" is the letter, return FAIL otherwise.',
        'expected_score': 0
    },
    {
        'description': 'Determining which is heavier between 2 tons of feathers and 1 ton of bricks. Pass case.',
        'input_text': '2 tons of feathers are heavier than 1 ton of bricks.',
        'target_text': 'Return PASS if the answer says that the feathers are heavier and does not at any point say they weigh the same amount, return FAIL otherwise.',
        'expected_score': 1
    },
    {
        'description': 'Determining which is heavier between 2 tons of feathers and 1 ton of bricks. Fail case.',
        'input_text': '2 tons of feathers and 1 ton of bricks weigh the same.',
        'target_text': 'Return PASS if the answer says that the feathers are heavier and does not at any point say they weigh the same amount, return FAIL otherwise.',
        'expected_score': 0
    },
    {
        'description': 'Solving the teleportation logic problem involving Doom Slayer and his companions. Pass case.',
        'input_text': 'Teleport with the Cacodemon, then teleport with the Bunny. Return with the Cacodemon, teleport with the Scientist, and finally teleport with the Cacodemon.',
        'target_text': 'Return PASS if the answer contains the following steps in this order: 1) Teleport with the Cacodemon, 2) Teleport with the Bunny, 3) Return with the Cacodemon, 4) Teleport with the Scientist, 5) Teleport with the Cacodemon. It may also include "teleport alone" steps, return FAIL otherwise.',
        'expected_score': 1
    },
    {
        'description': 'Solving the teleportation logic problem involving Doom Slayer and his companions. Fail case.',
        'input_text': 'Teleport with the Bunny, then teleport with the Cacodemon. Return with the Bunny, teleport with the Scientist, and finally teleport with the Cacodemon.',
        'target_text': 'Return PASS if the answer contains the following steps in this order: 1) Teleport with the Cacodemon, 2) Teleport with the Bunny, 3) Return with the Cacodemon, 4) Teleport with the Scientist, 5) Teleport with the Cacodemon. It may also include "teleport alone" steps, return FAIL otherwise.',
        'expected_score': 0
    }
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

async def run_all_examples():
    tasks = [evaluate_example(example) for example in test_cases]
    return await asyncio.gather(*tasks)

def main():
    print("Running all examples...\n")
    results = asyncio.run(run_all_examples())

    summary = {"pass": 0, "fail": 0}

    print("\nPerformance Report:")
    logging.info(f"Detailed Performance Report for PromptEvaluator:")
    for i, (example, (test_passed, model_result, expected)) in enumerate(zip(test_cases, results), 1):
        test_result = 'PASS' if test_passed else 'FAIL'
        summary[test_result.lower()] += 1

        report = (
            f"Example {i}:\n"
            f"  Description: {example['description']}\n"
            f"  Input: '{example['input_text']}'\n"
            f"  Target: '{example['target_text']}'\n"
            f"  Model Result: {model_result}\n"
            f"  Test Result: {test_result} (Expected: {'PASS' if expected == 1 else 'FAIL'})\n"
        )

        print(report)
        logging.info(report)

    summary_report = (
        "\nSummary:\n"
        f"  Total Examples: {len(test_cases)}\n"
        f"  Passed: {summary['pass']}\n"
        f"  Failed: {summary['fail']}\n"
    )

    print(summary_report)
    logging.info(summary_report)

if __name__ == "__main__":
    main()
