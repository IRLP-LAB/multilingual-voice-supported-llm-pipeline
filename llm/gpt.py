import openai

openai.api_key = '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'

def qa_system(question, context):
    response = openai.Completion.create(
        engine='text-davinci-003',  # Choose an appropriate OpenAI model
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Example usage
context = "Act like a indian dhabawala and answer the question asked"
answer = qa_system("What is the capital of France?", context)