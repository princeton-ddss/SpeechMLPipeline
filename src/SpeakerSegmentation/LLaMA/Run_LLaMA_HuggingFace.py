from LLaMA_HuggingFace import run_LLaMA_from_huggingface



# Prompting
systemprompt = """You are a helpful, respectful, and honest local food recommender at NYC. Your answer should only answer the question
once. You answer should not have any text both before and after the answer is done.
Please always answer the question as concise as possible without repeating previous text"""
question1 = "Could you list the two dessert shops at NYC I might like in terms of their name, location, and dessert name?"
answer1 = """Here is my answer: The first shop - name: Momofuku Milk Bar, location: East Village, dessert name: crack pie; 
The second shop - name: Levian Bakery, location: Upper West Side, dessert name: Chocolate Chip Cookies."""
questions_answers_dict = {question1: answer1}
main_question = "The answer is great! Could you list the other two dessert shops at NYC I might like by using the exact same format as your previous answer?"

output_path = "/Users/jf3375/PycharmProjects/SpeechMLPipeline/output/LLaMA"
output_filename = "llama_output.txt"

# LLaMA Model Specification
llama_model = "meta-llama/Llama-2-7b-chat-hf"

run_LLaMA_from_huggingface(systemprompt, main_question, output_path, output_filename,
                           questions_answers_dict=None,
                           llama_model=llama_model,
                           temperature=0.1, top_p=0, top_k=1, max_length=512)