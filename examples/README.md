
# Examples

This directory contains examples and demonstrations showcasing how to use the `PromptEvaluator` and `FactComparator` classes with different models and configurations.

## Running the Examples

Before running the examples, make sure you have set up the required dependencies and configured your model according to the instructions in the main project README.

### Usage

You can specify the model you want to use by passing it as an argument when running the script. The environment variable `INSPECT_MODEL_NAME` can be set to define the model to use. If not specified, it defaults to `openai/gpt-4`.

#### Example

```bash
cd examples
python prompt_evaluator_examples.py --model openai/gpt-3.5-turbo
python fact_comparator_examples.py --model openai/gpt-3.5-turbo
```

### Purpose

These scripts are specifically designed for seeing how well a particular model performs at evaluation, using these scorers. They do not run the full task process where a model is asked questions, and that input is evaluated. Instead, we provide specific text inputs, use an evaluation model, and compare the model evaluation outputs to the desired outputs. 

### Scorers

#### PromptEvaluator

The `PromptEvaluator` evaluates whether the provided input text meets certain conditions defined in the target text.

**Example Test Case:**

```json
{
    "description": "Checking the number of times the letter 'r' appears in the word 'carryforward'. Pass case.",
    "input_text": "The letter r appears four times in the word carryforward.",
    "target_text": "Return PASS if the answer says that the letter r is in the word carryforward four times, return FAIL otherwise.",
    "expected_score": 1
}
```

**Explanation:** In this test case, the input text is evaluated against the target text's condition. The input text says that the letter r appears four times. This is what the target_text, or prompt, is looking for. We're evaluating for whether the model, in conjunction with the scorer, returns a 1. If it does, it passes. 

#### FactComparator

The `FactComparator` compares facts between a context (target) and an answer (input). It evaluates the groundedness and thoroughness of the input based on the overlap between these two texts.

**Example Test Case:**

```json
{
    "description": "This is a basic use case with pronouns and mild rephrasing.",
    "input": "The Sun is a medium-sized star. It's about 4.6 billion years old.",
    "target": "The sun is approximately 4.6 billion years old. It's a mid-sized star.",
    "true_metrics": {"groundedness": 100, "thoroughness": 100}
}
```

**Explanation:** In this test case, the the input text contains two facts and those are the same facts as in the target text. Because of this, the desired behavior from the scorers is to return groundedness and thoroughness measures of 100. If it returns these, it passes.

These examples are not tests of the code running, but rather tests of how well the model is performing in conjunction with the scorers.
