import argparse
import asyncio
import os
import logging
from datetime import datetime

# Set up argument parsing
parser = argparse.ArgumentParser(description="Run FactComparator examples.")
parser.add_argument('--model', type=str, default='openai/gpt-4', help='The model name to use for evaluation.')
args = parser.parse_args()

# Set environment variable for model name
os.environ['INSPECT_MODEL_NAME'] = args.model

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging with a timestamped filename
log_filename = f"logs/fact_comparator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

from inspect_ai_scorers.code_from_inspect_ai import InspectChatModel
from inspect_ai_scorers.fact_comparator import FactComparator

cases = {
    'case1': {
        'input': 'The Sun is a medium-sized star. It\'s about 4.6 billion years old.',
        'target': 'The sun is approximately 4.6 billion years old. It\'s a mid-sized star.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},
        'description': 'This is a basic use case with pronouns and mild rephrasing.'
    },
    'case2': {
        'input': 'The Sun is a star. It is at the center of our Solar System and is 4.6 billion years old.',
        'target': 'The sun is at the center of our Solar System and is approximately 4.6 billion years old.',
        'true_metrics': {'groundedness': 67, 'thoroughness': 67},
        'description': 'This is a use case with partial fact overlap.'
    },
    'case3': {
        'input': 'Rachel owns a cat named Sally.',
        'target': 'Sally is a cat. Rachel is her owner.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},
        'description': 'This case involves simple restructuring and clarification.'
    },
    'case4': {
        'input': 'Stan is smaller than Sally.',
        'target': 'Sally is larger than Stan.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},
        'description': 'This case demonstrates a change in comparative perspective.'
    },
    'case5': {
        'input': 'The temperature today is 20 degrees Celsius.',
        'target': 'The average temperature today is 20 degrees Celsius.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},
        'description': 'This case involves exact matching of temperature with slight rephrasing.'
    },
    'case6': {
        'input': 'The temperature today is 20 degrees Celsius.',
        'target': 'The average temperature today is 50 degrees Celsius.',
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},
        'description': 'This case involves incorrect temperature information.'
    },
    'case7': {
        'input': 'The company has an ATO now, so they have been sanctioned by the government and you can work with them.',
        'target': 'The company has been sanctioned by the government in response to recent lawbreaking activity.',
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},
        'description': 'This case uses "sanctioned" in a way that highlights its dual meaning: approved or penalized.'
    },
    'case8': {
        'input': 'John is taller than Mike. John is older than Mike.',
        'target': 'Mike is shorter than John.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 50},
        'description': 'This case demonstrates comparative statements with additional facts.'
    },
    'case9': {
        'input': 'Alice went to the market. She bought apples, oranges, and bananas.',
        'target': 'Alice bought apples and oranges at the market.',
        'true_metrics': {'groundedness': 67, 'thoroughness': 100},
        'description': 'This case involves a partial list of items bought at the market.'
    },
    'case10': {
        'input': 'The car is red and fast.',
        'target': 'The car is red.',
        'true_metrics': {'groundedness': 50, 'thoroughness': 100},
        'description': 'This case involves partial attribute description.'
    },
    'case11': {
        'input': 'The city of Paris is in France.',
        'target': 'Paris is a city in France. It is known for the Eiffel Tower.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 50},
        'description': 'This case involves geographic location with additional context in the target.'
    },
    'case12': {
        'input': 'The library opens at 9 AM. The museum opens at 10 AM.',
        'target': 'The library opens at 9 AM.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 50},
        'description': 'This case involves different opening times for different places.'
    },
    'case13': {
        'input': 'Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.',
        'target': 'Water boils at 100 degrees Celsius.',
        'true_metrics': {'groundedness': 50, 'thoroughness': 100},
        'description': 'This case involves boiling and freezing points of water.'
    }
}

async def evaluate_case(case):
    context_text = case['target']
    answer_text = case['input']
    true_metrics = case['true_metrics']

    model_comparator = FactComparator(InspectChatModel())
    
    try:
        result = await model_comparator(context_text, answer_text)
        metrics = model_comparator.calculate_metrics(result["comparison_result"])

        groundedness_model = round(metrics['groundedness'])
        thoroughness_model = round(metrics['thoroughness'])

        groundedness_test_passed = abs(groundedness_model - true_metrics['groundedness']) < 1
        thoroughness_test_passed = abs(thoroughness_model - true_metrics['thoroughness']) < 1

        test_passed = groundedness_test_passed and thoroughness_test_passed
        model_error = None
    except Exception as e:
        groundedness_model = None
        thoroughness_model = None
        test_passed = False
        model_error = str(e)

    return {
        'Groundedness (Model)': groundedness_model,
        'Thoroughness (Model)': thoroughness_model,
        'Groundedness (Expected)': true_metrics['groundedness'],
        'Thoroughness (Expected)': true_metrics['thoroughness'],
        'Test Passed': test_passed,
        'Model Error': model_error,
        'Raw Result': result["comparison_result"] if 'result' in locals() else None  # Include raw result for debugging
    }

async def run_all_cases():
    tasks = [evaluate_case(case) for case in cases.values()]
    return await asyncio.gather(*tasks)

def main():
    print("Running all cases...\n")
    results = asyncio.run(run_all_cases())

    summary = {"pass": 0, "fail": 0}

    print("\nPerformance Report:")
    logging.info(f"Detailed Performance Report for FactComparator:")
    for i, (case_name, result) in enumerate(zip(cases.keys(), results), 1):
        case = cases[case_name]
        test_passed = result['Test Passed']
        summary["pass" if test_passed else "fail"] += 1

        report = (
            f"Case {i}: {case_name}\n"
            f"  Description: {case['description']}\n"
            f"  Input: '{case['input']}'\n"
            f"  Target: '{case['target']}'\n"
            f"  Groundedness (Model): {result['Groundedness (Model)']}\n"
            f"  Thoroughness (Model): {result['Thoroughness (Model)']}\n"
            f"  Groundedness (Expected): {result['Groundedness (Expected)']}\n"
            f"  Thoroughness (Expected): {result['Thoroughness (Expected)']}\n"
            f"  Test Passed: {'PASS' if test_passed else 'FAIL'}\n"
            f"  Model Error: {result['Model Error'] if result['Model Error'] else 'None'}\n"
            f"  Raw Result: {result['Raw Result']}\n"
        )

        print(report)
        logging.info(report)

    summary_report = (
        "\nSummary:\n"
        f"  Total Cases: {len(cases)}\n"
        f"  Passed: {summary['pass']}\n"
        f"  Failed: {summary['fail']}\n"
    )

    print(summary_report)
    logging.info(summary_report)

if __name__ == "__main__":
    main()
