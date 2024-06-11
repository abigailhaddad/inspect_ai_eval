import asyncio
import os
import logging
from datetime import datetime

os.environ['INSPECT_EVAL_MODEL'] = 'openai/gpt-4'
os.environ['INSPECT_MODEL_NAME'] = 'openai/gpt-4'

from new_scorers.code_from_inspect_ai import InspectChatModel
from new_scorers.fact_comparator import FactComparator

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging with a timestamped filename
log_filename = f"logs/fact_comparator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

cases = {
    'case1': {
        'input': 'The Sun is a medium-sized star. It\'s about 4.6 billion years old.',
        'target': 'The sun is approximately 4.6 billion years old. It\'s a mid-sized star.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},
        'description': 'This is a basic use case with pronouns and mild rephrasing.'
    },
    'case2': {
        'input': 'The Sun, a medium-sized star, is located at the center of our Solar System and is approximately 4.6 billion years old.',
        'target': 'The sun is a mid-sized star which has existed for about 4.6 billion years.',
        'true_metrics': {'groundedness': 67, 'thoroughness': 100},
        'description': 'This is a basic use case with mild rephrasing.'
    },
    'case3': {
        'input': 'Sally is Rachel\'s cat.',
        'target': 'Sally is a cat. Rachel is her owner.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},  
        'description': 'This case involves simple restructuring and clarification.'
    },
    'case4': {
        'input': 'Sally is larger than Stan.',
        'target': 'Stan is smaller than Sally.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100}, 
        'description': 'This case demonstrates a change in comparative perspective.'
    },
    'case5': {
        'input': 'the average temperature today is 20 degrees celsius.',
        'target': 'the mean temperature today is 68 degrees fahrenheit.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},  
        'description': 'This case involves unit conversion and synonym use.'
    },
    'case6': {
        'input': 'the average temperature today is 20 degrees celsius.',
        'target': 'the average temperature today is 50 degrees celsius.',
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},  
        'description': 'This case involves unit conversion and synonym use.'
    },
    'case7': {
        'input': 'The company has an ATO now, so they have been sanctioned by the government and you can work with them.', 
        'target':  'The company has been sanctioned by the government in response to recent lawbreaking activity.' , 
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},  # Contextual misuse
        'description': 'This case uses "sanctioned" in a way that highlights its dual meaning: approved or penalized.'
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
