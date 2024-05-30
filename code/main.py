

import inspect_ai_integration
import code_from_inspect_ai
import os
from inspect_ai import eval, Task, task

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
    
# Example usage
cases = {
    'case1': {
        'input': 'The Sun is a medium-sized star. It\'s about 4.6 billion years old.',
        'target': 'The sun is approximately 4.6 billion years old. It\'s a mid-sized star.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},
        'description': 'This is a basic use case with pronouns and mild rephrasing.'
    },
    'case2': {
        'input': 'The Sun, a medium-sized star, is located at the center of our Solar System and is approximately 4.6 billion years old.',
        'target': 'The sun is a mid-sized star which has existed for about 4.6 billion years.',
        'true_metrics': {'groundedness': 67, 'thoroughness': 100},
        'description': 'This is a basic use case with mild rephrasing.'
    },
    'case3': {
        'input': 'Sally is Rachel\'s cat.',
        'target': 'Sally is a cat. Rachel is her owner.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},  
        'description': 'This case involves simple restructuring and clarification.'
    },
    'case4': {
        'input': 'Sally is larger than Stan.',
        'target': 'Stan is smaller than Sally.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100}, 
        'description': 'This case demonstrates a change in comparative perspective.'
    },
    'case5': {
        'input': 'the average temperature today is 20 degrees celsius.',
        'target': 'the mean temperature today is 68 degrees fahrenheit.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},  
        'description': 'This case involves unit conversion and synonym use.'
    },
    'case6': {
        'input': 'the average temperature today is 20 degrees celsius.',
        'target': 'the average temperature today is 50 degrees celsius.',
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},  
        'description': 'This case involves unit conversion and synonym use.'
    },
    'case7': {
        'input': 'The company has an ATO now, so they have been sanctioned by the government and you can work with them.', 
        'target':  'The company has been sanctioned by the government in response to recent lawbreaking activity.' , 
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},  # Contextual misuse
        'description': 'This case uses "sanctioned" in a way that highlights its dual meaning: approved or penalized.'
    }}
    # Add more cases as needed

# Example usage
cases = {
    'case1': {
        'input': 'The Sun is a medium-sized star. It\'s about 4.6 billion years old.',
        'target': 'The sun is approximately 4.6 billion years old. It\'s a mid-sized star.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},
        'description': 'This is a basic use case with pronouns and mild rephrasing.'
    },
    'case2': {
        'input': 'The Sun, a medium-sized star, is located at the center of our Solar System and is approximately 4.6 billion years old.',
        'target': 'The sun is a mid-sized star which has existed for about 4.6 billion years.',
        'true_metrics': {'groundedness': 67, 'thoroughness': 100},
        'description': 'This is a basic use case with mild rephrasing.'
    },
    'case3': {
        'input': 'Sally is Rachel\'s cat.',
        'target': 'Sally is a cat. Rachel is her owner.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},  
        'description': 'This case involves simple restructuring and clarification.'
    },
    'case4': {
        'input': 'Sally is larger than Stan.',
        'target': 'Stan is smaller than Sally.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100}, 
        'description': 'This case demonstrates a change in comparative perspective.'
    },
    'case5': {
        'input': 'the average temperature today is 20 degrees celsius.',
        'target': 'the mean temperature today is 68 degrees fahrenheit.',
        'true_metrics': {'groundedness': 100, 'thoroughness': 100},  
        'description': 'This case involves unit conversion and synonym use.'
    },
    'case6': {
        'input': 'the average temperature today is 20 degrees celsius.',
        'target': 'the average temperature today is 50 degrees celsius.',
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},  
        'description': 'This case involves unit conversion and synonym use.'
    },
    'case7': {
        'input': 'The company has an ATO now, so they have been sanctioned by the government and you can work with them.', 
        'target':  'The company has been sanctioned by the government in response to recent lawbreaking activity.' , 
        'true_metrics': {'groundedness': 0, 'thoroughness': 0},  # Contextual misuse
        'description': 'This case uses "sanctioned" in a way that highlights its dual meaning: approved or penalized.'
    },
    # Add more cases as needed
}



if __name__ == "__main__":
    import asyncio
    asyncio.run(smallest_test())
    df = inspect_ai_integration.compare_metrics(cases)
    print(df)
    eval(inspect_ai_integration.my_eval(), model="openai/gpt-3.5-turbo")
