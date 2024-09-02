from inspect_ai.model import get_model
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer


@scorer(metrics=[accuracy()])
def prompt_scorer(model) -> Scorer:
    """
    Create a scorer for the prompt evaluator.

    Args:
        model: The AI model used for evaluating prompts.

    Returns:
        Scorer: The prompt evaluator scorer.
    """
    # resolve model
    grader_model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        
        # The grading prompt
        prompt = """
Return PASS if the 'Answer' below fulfills the requirements outlined in the 'Target' below, otherwise return FAIL:

Answer:
{answer}

Target:
{target}
"""
        # format the prompt
        prompt = prompt.format(
                answer=state.output.completion, target=target.target[0])

        # generate the completion
        result = await grader_model.generate(prompt)
        final_result = result.completion

        # compute the score
        if "PASS" in final_result:
            pass_value = "C"
        elif "FAIL" in final_result:
            pass_value = "I"
        else:
            pass_value = "I"
        
        return Score(
            value=pass_value,
            answer=final_result
        )

    return score

