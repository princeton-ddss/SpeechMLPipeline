'''
Run LLaMA for the Speaker Segmentation and Speaker Diarization
'''

#!/usr/bin/env python
# coding: utf-8

import transformers
import torch

offline = False

# 7b is the smallest model; 7b vs 13b vs 70b
llama_model = "meta-llama/Llama-2-7b-chat-hf"

# Function to formulate the full prompt for LLaMa
def get_full_prompt(systemprompt,  main_question, questions_answers_dict = None,
                    INST = '[INST]', E_INST = '[/INST]',
                    SYS = '<<SYS>>\n', E_SYS = '\n<</SYS>>\n\n',
                    SEN = '<s>', EN_SEN = '<\s>'):
    if not questions_answers_dict: # No few-shot learning
        full_prompt =   SEN + INST + \
                        SYS + systemprompt + E_SYS + \
                        main_question + E_INST
    else:
        questions = list(questions_answers_dict.keys())
        full_prompt =   SEN + INST + \
                        SYS + systemprompt + E_SYS + \
                        questions[0] + E_INST + \
                        questions_answers_dict[questions[0]] + EN_SEN

        for question in questions[1:]:
            full_prompt += SEN + INST + question + E_INST + questions_answers_dict[question] + EN_SEN

        full_prompt += SEN + INST + main_question + E_INST
    return full_prompt

def run_LLaMA_from_huggingface(systemprompt, main_question, output_path, output_filename,
                               questions_answers_dict = None,
                               llama_model = "meta-llama/Llama-2-7b-chat",
                               temperature = 0.1, top_p = 0, top_k = 1, max_length = 512):
    print('Use LLaMA via HuggingFace')
    print('Make sure to log into the huggingface account via terminal: huggingface-cli login')
    # Import Tokenizer and Pipeline
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model,
                                                           return_token_type_ids=False)
    pipeline = transformers.pipeline(
        "text-generation",
        model=llama_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Run LLaMa
    full_prompt = get_full_prompt(systemprompt, main_question, questions_answers_dict)
    print(full_prompt)

    # Set Options to Get Deterministic Response
    sequences = pipeline(
        full_prompt,
        return_full_text = False, # Not repeat the question
        do_sample= True,
        top_p = top_p,
        top_k = top_k,
        num_return_sequences = 1,
        eos_token_id = tokenizer.eos_token_id,
        max_length = max_length,
        temperature = temperature # Currently does not support the set of 0
    )
    print('----------------------------------------------------------------')
    print('Answer Started')

    with open("{}/{}".format(output_path, output_filename), "w", newline="") as file:
        for seq in sequences:
            txt = seq['generated_text']
            print(f"Result: %{txt}")
            file.write(txt)