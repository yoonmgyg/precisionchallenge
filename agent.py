from smolagents import Agent
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # Replace with your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = llm_model.generate(inputs, max_length=256, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

agent = Agent(
    llm_model=generate_response,
    prompt_template=prompt_template,
    retrieval_function=lambda query: retrieve_guidance(query, index, embeddings, chunks, sentence_transformer_model)
)

user_query = "What are the key compliance steps for launching a new cosmetic product?"
response = agent.run(user_query)
print("Agent Response:", response)
