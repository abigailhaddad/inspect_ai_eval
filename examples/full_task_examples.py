import os
import sys
import argparse
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log._log import EvalLog, EvalResults, EvalSample
from inspect_ai.scorer._metric import Score
from inspect_ai._eval.eval import EvalLogs
from inspect_ai.solver import generate, system_message
from inspect_ai.model import get_model
from inspect_ai_scorers.code_from_inspect_ai import InspectChatModel
from inspect_ai_scorers.prompt_evaluator import prompt_scorer
from inspect_ai_scorers.fact_comparator import fact_comparator_scorer

@task
def prompt_evaluator_eval(eval_model):
    """
    Create an evaluation task for the PromptEvaluator.
     
    Args:
        eval_model: The AI model used for the evaluation task.
     
    Returns:
        Task: The evaluation task.
    """
    samples = [
        Sample(
            input="How old is the sun?",
            target="Return PASS if the answer contains that the sun is 4.6 billion years old, return FAIL otherwise.",
            description="Very basic question.",
            id="case1"
        )
    ]
    SYSTEM_MESSAGE = "Please answer the question being asked."
    return Task(
        dataset=samples,
        plan=[
            system_message(SYSTEM_MESSAGE),
            generate(),
        ],
        scorer=prompt_scorer(eval_model),
    )

@task
def fact_comparator_eval(eval_model):
    """
    Create an evaluation task for the fact comparator.
     
    Args:
        eval_model: The AI model used for the evaluation task.
     
    Returns:
        Task: The evaluation task.
    """
    samples = [
        Sample(
            input="How old is the sun?",
            target="The sun is approximately 4.6 billion years old. It's a mid-sized star.",
            description="Very basic question.",
            id="case1"
        )
    ]
    SYSTEM_MESSAGE = "Please answer the question being asked."
    return Task(
        dataset=samples,
        plan=[
            system_message(SYSTEM_MESSAGE),
            generate(),
        ],
        scorer=fact_comparator_scorer(eval_model),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations with specified models.")
    parser.add_argument("--eval_model", type=str, default='openai/gpt-4',
                        help="Model to use for evaluation (default: openai/gpt-4)")
    parser.add_argument("--query_model", type=str, default='openai/gpt-3.5-turbo',
                        help="Model to use for querying (default: openai/gpt-3.5-turbo)")
    args = parser.parse_args()

    os.environ['INSPECT_EVAL_MODEL'] = args.eval_model
    eval_model = InspectChatModel()
    query_model = args.query_model
    
    print(f"Using evaluation model: {args.eval_model}")
    print(f"Using query model: {args.query_model}")
    
    print("\nRunning prompt_evaluator_eval:")
    prompt_eval_results = eval(prompt_evaluator_eval(eval_model), model=query_model)
    print(prompt_eval_results)
    
    print("\nRunning fact_comparator_eval:")
    fact_eval_results = eval(fact_comparator_eval(eval_model), model=query_model)
    print(fact_eval_results)