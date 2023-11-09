'''
Interface to Run LLaMa from Github Repo Offline without specifying prompts on the command line

Usage: torchrun --nproc_per_node 1 Run_LLaMA_GithubRepo_Diarization_Basic.py
'''
import pandas as pd
import re

from LLaMA_GithubRepo import run_LLaMA_from_repo




def main():
    # Read the Audio-to-Text Transcription as Input
    # input_path = '/Users/jf3375/PycharmProjects/SpeechMLPipeline/output/Whisper'
    input_path = '/scratch/gpfs/jf3375/repos/llama-main/LLaMA'
    input_filename = 'family_sample.csv'
    text_df = pd.read_csv("{}/{}".format(input_path, input_filename))
    sentences = list(text_df['text'])

    for idx, sentence in enumerate(sentences):
        sentences[idx] = 'Sentence: {} \n'.format(sentence)


    # Prompting
    systemprompt = """You are a kindergarden teacher who talks a lot to chidren. 
    You are familiar with the daily life conversation between different family members.
    You are very familiar with the way children speaks.
    You are very proficient in the speaker segmentation and speaker diarization.
    Please double check the answer by carefully considering how children and parents speak differently.
    Your answer should only answer the question once. 
    Please always answer the question as concise as possible without repeating previous text. 
    Please double check to ensure that you identify the speaker of every sentence.
    Please explicitly put both speaker and speaker name in the answer.
    """
    # Need to format question and answer in a specific format
    question1 = """"Could you identify which family member spoke which sentence?
Sentence: What is the breakfast today?
Sentence: Omelete with ham and cheese
Sentence: That sounds delicious. 
Sentence: I also want some orange juice.
Sentence: Could you eat faster? We need to hurry to go to school.
Sentence: Sure
    """
    systemprompt = systemprompt.replace("\n", " ")
    answer1 = """Here is my answer: 
Sentence - speaker: kid;
Sentence - speaker: parents; 
Sentence - speaker: kid; 
Sentence - speaker: kid; 
Sentence - speaker: parents; 
Sentence - speaker: parents."""
    main_question = """The answer is great! Could you identify which family member spoke which sentence by using the exact same format as your previous answer?\n"""
    for sentence in sentences:
        main_question += sentence
    print(main_question)

    questions_answers_dict = {question1: answer1}
    output_path = "/scratch/gpfs/jf3375/tempoutput/llama"
    output_filename = "llama_diarization.txt"
    output_csv_filename = "llama_diarization.csv"

    # Specify Model Hyperparameters
    # Need to increase sequence length so it returns all the answers
    # The sequence length includes the initial prompt
    ckpt_dir =  '/scratch/gpfs/jf3375/repos/llama-main/llama-2-7b-chat/'
    tokenizer_path = '/scratch/gpfs/jf3375/repos/llama-main/tokenizer.model'
    temperature = 0.1
    top_p = 0
    max_seq_len = 2000
    max_batch_size = 8
    max_gen_len  = None

    run_LLaMA_from_repo(systemprompt,  main_question, output_path, output_filename, questions_answers_dict,
                        ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len)

    speakers = [0]*len(sentences)
    idx = 0
    with open('{}/{}'.format(output_path, output_filename)) as f:
        lines = f.readlines()
        for line in lines:
            line = line.lower()
            if 'speaker' in line:
                speaker = re.sub(r'[^\w]', ' ', line.split('speaker')[1]).strip()
                speakers[idx] = speaker
                idx += 1
    if 0 in speakers:
        raise Exception('Does not identify speaker of each sentence')
    text_df['speaker_predict'] = speakers
    text_df.to_csv('{}/{}'.format(output_path, output_csv_filename), index=False)


if __name__ == "__main__":
    main()