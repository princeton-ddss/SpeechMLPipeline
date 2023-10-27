'''
Interface to Run LLaMa from Github Repo Offline without specifying prompts on the command line

Usage: torchrun --nproc_per_node 1 Run_LLaMA_GithubRepo.py
'''



from LLaMA_GithubRepo import run_LLaMA_from_repo
def main():
    # Prompting
    systemprompt = """You are a helpful, respectful, and honest local food recommender at NYC. Your answer should only answer the question
    once. You answer should not have any text both before and after the answer is done.
    Please always answer the question as concise as possible without repeating previous text"""
    question1 = "Could you list the two dessert shops at NYC I might like in terms of their name, location, and dessert name?"
    answer1 = """Here is my answer: The first shop - name: Momofuku Milk Bar, location: East Village, dessert name: crack pie; 
    The second shop - name: Levian Bakery, location: Upper West Side, dessert name: Chocolate Chip Cookies."""
    questions_answers_dict = {question1: answer1}
    main_question = "The answer is great! Could you list the other two dessert shops at NYC I might like by using the exact same format as your previous answer?"
    output_path = "/scratch/gpfs/jf3375/tempoutput/llama"
    output_filename = "llama_sample.txt"

    # Specify Model Hyperparameters
    ckpt_dir =  '../llama-2-7b-chat/'
    tokenizer_path = '../tokenizer.model'
    temperature = 0.1
    top_p = 0
    max_seq_len = 512
    max_batch_size = 8
    max_gen_len  = None

    run_LLaMA_from_repo(systemprompt,  main_question, output_path, output_filename, questions_answers_dict,
                        ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len)

if __name__ == "__main__":
    main()