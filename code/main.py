

import inspect_ai_integration
import code_from_inspect_ai
import os

# Set environment variables
os.environ['INSPECT_EVAL_MODEL'] = 'openai/gpt-4'
os.environ['INSPECT_MODEL_NAME'] = 'openai/gpt-4'

async def smallest_test():
    # Create an instance of InspectChatModel with the specified model
    inspect_model = code_from_inspect_ai.InspectChatModel()

    # Create an instance of FactComparator with the InspectChatModel
    comparator = inspect_ai_integration.FactComparator(inspect_model)

    context = "The fox is brown. It runs quickly. The fox's best friend is Sally, which is a cat."
    answer = "The fox is tan. It runs fast. Its best friend is a cat. She's named Sally."

    # Run the asynchronous process_data method
    result = await comparator(context, answer)

    metrics = comparator.calculate_metrics(result["comparison_result"])

    print("\nContext list:")
    print(result["context_list"])

    print("\nAnswer list:")
    print(result["answer_list"])

    print("\nComparison result:")
    print(result["comparison_result"])

    print("\nMetrics:")
    print(f"Groundedness: {metrics['groundedness']:.2f}%")
    print(f"Thoroughness: {metrics['thoroughness']:.2f}%")

# To run the async function from the top-level script
if __name__ == "__main__":
    import asyncio
    asyncio.run(smallest_test())
