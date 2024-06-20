
# Examples

This directory contains examples and demonstrations showcasing how to use the `PromptEvaluator` and `FactComparator` classes. 

There are two ways to run these classes: you can run it with the text you are evaluating already generated, like in the fact_comparator_examples and prompt_evaluator_examples. In these examples, we know what we expect the answers to be, so you can also use this as a way to test out different evaluation models.

You can also run the full task, where you fetch responses from an LLM to evaluate and then you run the evaluations, as in the full_task_examples. In these examples, we don't know what the correct answer should be, because we don't know what the initial LLM is going to return. 


## Running the Examples

Before running the examples, make sure you have set up the required dependencies and configured your model according to the instructions in the main project README.


### Usage

You can specify the models you want to use by passing them as an argument when running the script or you can use the defaults. If not specified, the evaluation model defaults to `openai/gpt-4` and the model we're evaluating defaults to `openai/gpt-3.5-turbo` (for the full_task_examples).

You can specify other models to use like below:


#### Example

```bash
cd examples
python prompt_evaluator_examples.py --model openai/gpt-3.5-turbo
python fact_comparator_examples.py --model openai/gpt-3.5-turbo
python full_task_examples.py --eval_model anthropic/claude-2.1 --query_model openai/gpt-3.5-turbo
```

### Purpose

The prompt_evaluator_examples and fact_comparator_examples are specifically designed for seeing how well a particular model performs at evaluation, using these scorers. They do not run the full task process where a model is asked questions, and that input is evaluated. Instead, we provide specific text inputs, use an evaluation model, and compare the model evaluation outputs to the desired outputs. 

The full_task_examples show how to use the scorers as part of a full task, where a model is asked questions and the input is evaluated. 

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




Certainly! I'll add a section for full_task_examples to the README. Here's the updated version with the new section added:

```markdown
# Examples

This directory contains examples and demonstrations showcasing how to use the `PromptEvaluator` and `FactComparator` classes. 

There are two ways to run these classes: you can run it with the text you are evaluating already generated, like in the fact_comparator_examples and prompt_evaluator_examples. In these examples, we know what we expect the answers to be, so you can also use this as a way to test out different evaluation models.

You can also run the full task, where you fetch responses from an LLM to evaluate and then you run the evaluations, as in the full_task_examples. In these examples, we don't know what the correct answer should be, because we don't know what the initial LLM is going to return. 

## Running the Examples

Before running the examples, make sure you have set up the required dependencies and configured your model according to the instructions in the main project README.

### Usage

You can specify the models you want to use by passing them as an argument when running the script or you can use the defaults. If not specified, the evaluation model defaults to `openai/gpt-4` and the model we're evaluating defaults to `openai/gpt-3.5-turbo` (for the full_task_examples).

You can specify other models to use like below:

#### Example

```bash
cd examples
python prompt_evaluator_examples.py --model openai/gpt-3.5-turbo
python fact_comparator_examples.py --model openai/gpt-3.5-turbo
python full_task_examples.py --eval_model anthropic/claude-2.1 --query_model openai/gpt-3.5-turbo
```

### Purpose

The prompt_evaluator_examples and fact_comparator_examples are specifically designed for seeing how well a particular model performs at evaluation, using these scorers. They do not run the full task process where a model is asked questions, and that input is evaluated. Instead, we provide specific text inputs, use an evaluation model, and compare the model evaluation outputs to the desired outputs. 

The full_task_examples show how to use the scorers as part of a full task, where a model is asked questions and the input is evaluated. 

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

**Explanation:** In this test case, the input text contains two facts and those are the same facts as in the target text. Because of this, the desired behavior from the scorers is to return groundedness and thoroughness measures of 100. If it returns these, it passes.

These examples are not tests of the code running, but rather tests of how well the model is performing in conjunction with the scorers.

### Full Task Examples

The `full_task_examples.py` script demonstrates how to use both the `PromptEvaluator` and `FactComparator` in a complete evaluation task. This example shows:

1. How to set up evaluation and query models separately.
2. How to create evaluation tasks for both `PromptEvaluator` and `FactComparator`.
3. How to run evaluations using the specified models.

**Usage:**

```bash
python full_task_examples.py --eval_model openai/gpt-4 --query_model openai/gpt-3.5-turbo
```

In this example:
- The `eval_model` is used to evaluate the responses (default: openai/gpt-4).
- The `query_model` is used to generate responses to the input questions (default: openai/gpt-3.5-turbo).

The script runs two separate evaluations:

1. `prompt_evaluator_eval`: Uses the `PromptEvaluator` to check if the generated answer meets specific criteria.
2. `fact_comparator_eval`: Uses the `FactComparator` to compare the generated answer against a set of known facts.

This full task example allows you to see how the query model performs in generating answers and how well the evaluation model assesses those answers using different evaluation criteria.
```