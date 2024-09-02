from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message
from inspect_ai_scorers._fact_scorer import fact_scorer
from inspect_ai_scorers.prompt_evaluator import prompt_scorer


@task
def prompt_evaluator_eval():
    samples = [
        Sample(
            input="Teleport with the Cacodemon, then teleport with the Bunny. Return with the Cacodemon, teleport with the Scientist, and finally teleport with the Cacodemon.",
            target="the following steps in this order: 1) Teleport with the Cacodemon, 2) Teleport with the Bunny, 3) Return with the Cacodemon, 4) Teleport with the Scientist, 5) Teleport with the Cacodemon. It may also include 'teleport alone' steps",
            description="Very basic question.",
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
        ),
        Sample(input="Please describe the steps to boil a pot of water.",
               target="Get a pot. Fill pot with water. Place pot on stove. Turn on stove. Heat until water is boiling."
               )
    ]
    SYSTEM_MESSAGE = "Please answer the question being asked."
    return Task(
        dataset=samples,
        plan=[
            system_message(SYSTEM_MESSAGE),
            generate(),
        ],
        scorer=fact_scorer("openai/gpt-4"),
    )