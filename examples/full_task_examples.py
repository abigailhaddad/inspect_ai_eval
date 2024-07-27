from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message
from inspect_ai_scorers.prompt_evaluator import prompt_scorer
from inspect_ai_scorers.fact_comparator import fact_comparator_scorer


@task
def prompt_evaluator_eval():
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
        scorer=prompt_scorer("openai/gpt-4")
    )

@task
def fact_comparator_eval():
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
        scorer=fact_comparator_scorer("openai/gpt-4"),
    )