from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Score, Scorer, Target, metric, scorer
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage

from new_scorers.code_from_inspect_ai import InspectChatModel

class FactComparator:
    """
    A class to compare facts between context and answer using an AI model.
    """

    def __init__(self, model):
        """
        Initialize the FactComparator with the provided model.
        
        Args:
            model: The AI model used for generating and comparing facts.
        """
        self.model = model
        self.parser = PydanticOutputParser(pydantic_object=ComparisonResult)

    async def __call__(self, context_text, answer_text):
        """
        Process the context and answer asynchronously and return the comparison results.

        Args:
            context_text (str): The context text.
            answer_text (str): The answer text.

        Returns:
            dict: The comparison results.
        """
        return await self.process_data(context_text, answer_text)

    async def process_data(self, context_text, answer_text):
        """
        Process the context and answer, parsing them into facts and comparing.

        Args:
            context_text (str): The context text.
            answer_text (str): The answer text.

        Returns:
            dict: The processed data and comparison results.
        """
        context_list = (await self.model._agenerate([HumanMessage(content=self._parse_prompt().format(text=context_text))])).generations[0].text
        answer_list = (await self.model._agenerate([HumanMessage(content=self._parse_prompt().format(text=answer_text))])).generations[0].text
        comparison_result = self.parser.parse((await self.model._agenerate([HumanMessage(content=self._compare_prompt().format(context_list=context_list, answer_list=answer_list))])).generations[0].text)

        return {
            "context_list": context_list,
            "answer_list": answer_list,
            "comparison_result": comparison_result,
        }

    def calculate_metrics(self, comparison_result):
        """
        Calculate groundedness and thoroughness metrics based on the comparison results.

        Args:
            comparison_result (ComparisonResult): The result of the fact comparison.

        Returns:
            dict: The calculated metrics.
        """
        facts_in_both_count = len(comparison_result.facts_in_both)
        facts_only_in_answer_count = len(comparison_result.facts_only_in_answer)
        facts_only_in_context_count = len(comparison_result.facts_only_in_context)

        total_answer_facts = facts_in_both_count + facts_only_in_answer_count
        total_context_facts = facts_in_both_count + facts_only_in_context_count

        # Groundedness is the proportion of facts in the answer that are also in the context
        groundedness = (facts_in_both_count / total_answer_facts) * 100 if total_answer_facts > 0 else 0

        # Thoroughness is the proportion of facts in the context that are also in the answer
        thoroughness = (facts_in_both_count / total_context_facts) * 100 if total_context_facts > 0 else 0

        return {
            "groundedness": groundedness,
            "thoroughness": thoroughness,
        }

    @staticmethod
    def _parse_prompt():
        """
        Generate the prompt template for parsing facts from text.

        Returns:
            PromptTemplate: The prompt template.
        """
        return PromptTemplate(
            input_variables=["text"],
            template="""
            Here is a text that may contain one or more facts:

            <text>
            {text}
            </text>

            Please parse this text into a list of individual facts. If a sentence contains multiple facts, break it up into separate sentences as needed so that each sentence contains only one fact.

            If any of the facts contain pronouns and the pronoun reference is clear, replace the pronoun with the noun it refers to. If the pronoun reference is ambiguous, leave the pronoun as is.

            Return the final list of parsed and pronoun-replaced facts inside <facts> tags, with each fact on its own line. Do not include any additional commentary or explanation, including about pronoun changes, number of facts, or truth value of the facts.
        """,
        )

    @staticmethod
    def _compare_prompt():
        """
        Generate the prompt template for comparing facts between context and answer.

        Returns:
            PromptTemplate: The prompt template.
        """
        return PromptTemplate(
            input_variables=["context_list", "answer_list"],
            template="""
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
            """,
        )

class ComparisonResult(BaseModel):
    """
    A Pydantic model for representing the comparison result.
    """
    facts_in_both: list[str] = Field(default_factory=list, description="List of facts present in both context and answer")
    facts_only_in_answer: list[str] = Field(default_factory=list, description="List of facts only present in the answer")
    facts_only_in_context: list[str] = Field(default_factory=list, description="List of facts only present in the context")



class ModelComparator:
    """
    A class to compare models based on their generated facts.
    """

    def __init__(self, model):
        """
        Initialize the ModelComparator with the provided model.
        
        Args:
            model: The AI model used for generating and comparing facts.
        """
        self.inspect_model = InspectChatModel()
        self.comparator = FactComparator(self.inspect_model)

    async def run_and_compare(self, context_text, answer_text):
        """
        Run the model comparison and calculate the metrics.

        Args:
            context_text (str): The context text for comparison.
            answer_text (str): The answer text for comparison.

        Returns:
            dict: The comparison results and metrics.
        """
        try:
            result = await self.comparator(context_text, answer_text)
            metrics = self.comparator.calculate_metrics(result["comparison_result"])
            groundedness_model = metrics['groundedness']
            thoroughness_model = metrics['thoroughness']
            context_list = result["context_list"]
            answer_list = result["answer_list"]
            comparison_result = result["comparison_result"]
            model_error = None
        except Exception as e:
            groundedness_model = None
            thoroughness_model = None
            context_list = None
            answer_list = None
            comparison_result = None
            model_error = str(e)

        return {
            'Groundedness (Model)': groundedness_model,
            'Thoroughness (Model)': thoroughness_model,
            'Context List': context_list,
            'Answer List': answer_list,
            'Facts in Both': comparison_result.facts_in_both if comparison_result else None,
            'Facts Only in Answer': comparison_result.facts_only_in_answer if comparison_result else None,
            'Facts Only in Context': comparison_result.facts_only_in_context if comparison_result else None,
            'Model Error': model_error
        }


class FactComparatorScorer:
    """
    A class to score facts based on their groundedness and thoroughness.
    """

    def __init__(self, model):
        """
        Initialize the FactComparatorScorer with the provided model.
        
        Args:
            model: The AI model used for generating and comparing facts.
        """
        self.model = model
        self.fact_comparator = FactComparator(model)

    async def __call__(self, state: TaskState, target: Sample):
        """
        Process the state and target to calculate the score.

        Args:
            state (TaskState): The current task state.
            target (Sample): The target sample.

        Returns:
            Score: The calculated score.
        """
        try:
            context_text = state.output.choices[0].message.content
        except:
            context_text = state.input
        answer_text = target.target

        result = await self.fact_comparator.process_data(context_text, answer_text)
        metrics = self.fact_comparator.calculate_metrics(result["comparison_result"])

        scorer_value = {
            "groundedness": metrics["groundedness"],
            "thoroughness": metrics["thoroughness"],
        }

        explanation = str(result) + f"\nModel Output: {context_text}"

        return Score(
            value=scorer_value,
            explanation=explanation,
        )


@metric
def thoroughness():
    """
    Metric function to calculate the thoroughness score.

    Returns:
        function: The metric function.
    """
    def metric(scores: list[Score]) -> float:
        total = 0.0
        for item in scores:
            metadata = item.metadata
            if metadata is not None:
                total += float(metadata["thoroughness"])
        return total / float(len(scores))
    return metric


@metric
def groundedness():
    """
    Metric function to calculate the groundedness score.

    Returns:
        function: The metric function.
    """
    def metric(scores: list[Score]) -> float:
        total = 0.0
        for item in scores:
            metadata = item.metadata
            if metadata is not None:
                total += float(metadata["groundedness"])
        return total / float(len(scores))
    return metric


@scorer(metrics=[groundedness(), thoroughness()])
def fact_comparator_scorer(model) -> Scorer:
    """
    Create a scorer for the fact comparator.

    Args:
        model: The AI model used for generating and comparing facts.

    Returns:
        Scorer: The fact comparator scorer.
    """
    async def score(state: TaskState, target: Target) -> Score:
        model = InspectChatModel()
        fact_comparator_scorer = FactComparatorScorer(model)

        score = await fact_comparator_scorer(state, target)

        grounded_score = score.value['groundedness']
        thorough_score = score.value['thoroughness']
        explanation = score.explanation

        answer = state.output.completion

        return Score(
            value=f"G:{grounded_score} : T:{thorough_score}",
            answer=answer,
            explanation="nothing",
            metadata={
                "thoroughness": thorough_score,
                "groundedness": grounded_score,
                "stuff": explanation
            }
        )

    return score



