import json
from inspect_ai.model import Model, get_model
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr



fact_prompt = """
Here is a text that may contain one or more facts:

<text>
{text}
</text>

Please parse this text into a list of individual facts. If a sentence contains multiple facts, break it up into separate sentences as needed so that each sentence contains only one fact.

If any of the facts contain pronouns and the pronoun reference is clear, replace the pronoun with the noun it refers to. If the pronoun reference is ambiguous, leave the pronoun as is.

Return the final list of parsed and pronoun-replaced facts inside <facts> tags, with each fact on its own line. Do not include any additional commentary or explanation, including about pronoun changes, number of facts, or truth value of the facts.
"""

compare_prompt = """
You will be comparing facts between a context and an answer to determine which facts are shared and which are unique to each.

Here is the context:

<context>
{context_list}
</context>

And here is the answer: 

<answer>
{answer_list}
</answer>

Carefully analyze the facts presented in the context and answer, focusing on the semantic meaning rather than the exact wording.

Then, output a dictionary with the following keys and corresponding lists of facts as values:

1. "facts_in_both": A list of facts that are present in both the context and the answer

2. "facts_only_in_answer": A list of facts that are only present in the answer 

3. "facts_only_in_context": A list of facts that are only present in the context

Remember, the facts do not need to be worded identically to be considered the same. Focus on whether the core meaning is shared or unique.  A fact in the context may be expressed in different terms in the answer, or multiple facts in one may combine to express a single fact in the other.

Provide your results in this format:

{{
    "facts_in_both": [
        "Fact 1 present in both",
        "Fact 2 present in both"
    ],
    "facts_only_in_answer": [
        "Fact 1 only in answer",
        "Fact 2 only in answer"  
    ],
    "facts_only_in_context": [
        "Fact 1 only in context",
        "Fact 2 only in context"
    ]
}}
"""

explanation_format = """
Facts in Both:
{facts_in_both}

Facts only in Answer:
{facts_only_in_answer}

Facts only in Context:
{facts_only_in_context}
"""

@scorer(metrics={"groundedness": [mean(), stderr()], "thoroughness": [mean(), stderr()]})
def fact_scorer(model: str | Model | None = None) -> Scorer:
    
    # TODO: Could add an option to have a separate fact and grader model
    fact_model = get_model(model)
    grader_model = get_model(model)

    """
    Create a scorer for the fact comparator.

    Args:
        model: The AI model used for generating and comparing facts.

    Returns:
        Scorer: The fact comparator scorer.
    """
    async def score(state: TaskState, target: Target) -> Score:
        

        # First establish the facts in the target
        target_facts = await fact_model.generate(fact_prompt.format(text=target.target))

        # Now establish the facts in the answer
        answer_facts = await fact_model.generate(fact_prompt.format(text=state.output.completion))

        # Compare the facts
        compare_result = await grader_model.generate(compare_prompt.format(
            context_list = target_facts,
            answer_list = answer_facts
        ))

        # TODO: Validate this result parses
        comparison_result = json.loads(compare_result.completion)

        # Basic counts for computing values
        facts_in_both_count = len(comparison_result["facts_in_both"])
        facts_only_in_answer_count = len(comparison_result["facts_only_in_answer"])
        facts_only_in_context_count = len(comparison_result["facts_only_in_context"])

        total_answer_facts = facts_in_both_count + facts_only_in_answer_count
        total_context_facts = facts_in_both_count + facts_only_in_context_count

        # Groundedness is the proportion of facts in the answer that are also in the context
        groundedness = (facts_in_both_count / total_answer_facts) * 100 if total_answer_facts > 0 else 0

        # Thoroughness is the proportion of facts in the context that are also in the answer
        thoroughness = (facts_in_both_count / total_context_facts) * 100 if total_context_facts > 0 else 0

        explanation = explanation_format.format(
            facts_in_both = "\n".join(comparison_result["facts_in_both"]),
            facts_only_in_answer = "\n".join(comparison_result["facts_only_in_answer"]),
            facts_only_in_context = "\n".join(comparison_result["facts_only_in_context"])
          )
        answer = state.output.completion

        return Score(
            value={
                "groundedness": groundedness,
                "thoroughness": thoroughness
            },
            answer=answer,
            explanation=explanation,
        )

    return score