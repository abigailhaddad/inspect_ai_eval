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