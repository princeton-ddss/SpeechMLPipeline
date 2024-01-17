import json
import pandas as pd
import os

from LLaMA_HuggingFace_Offline import get_full_prompt
from LLaMA_HuggingFace_Offline import setup_LLaMA_model_tokenizer

from prompt1 import systemprompt, main_question_bgn, examples
from prompt1_template import get_main_question

# Clear Memory first
import torch
torch.cuda.empty_cache()

import gc
gc.collect()

# Need to shrink text to 4096-input text

# Set llama model directory
llama_model_dir = "/scratch/gpfs/jf3375/models/huggingface/hub/llama"

# Find the name of downloaded folder llama where config.json is
# llama_folder_name = 'models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8'
llama_folder_name = '/scratch/gpfs/jf3375/models/huggingface/hub/llama/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496'
# llama_folder_name = '/scratch/gpfs/jf3375/models/huggingface/hub/llama/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2'

#Use whatever gpus avaialble, suggested by hf
device_map = 'auto'
#32 torchdtype
torch_dtype = torch.float16

# Import LLaMA Tokenizer and Pipeline Offline
# One GPU: device = '0'
# Two GPUs: device ='0,1'
# Find the torch device 0 name
# torch.cuda.get_device_name(torch.device('cuda:0'))
# Check gpu resource
# nvidia-smi

#gou_num_devices
tokenizer, pipeline = setup_LLaMA_model_tokenizer(llama_model_dir, llama_folder_name, device_map, torch_dtype)

# Set parameters of the model to answer
# The input includes the question itself; Need to set up max_length long enough so all the answers would return
temperature = 0.1
top_p = 0
top_k = 1
max_length = 4086 #The maximum number of tokens; Issues would occur if it increases

whisper_output_path = '/scratch/gpfs/jf3375/modern_family/output/Whispertimestamped'
# whisper_output_path = '/Users/jf3375/Desktop/modern_family/output/Whispertimestamped'
input_filename = 'sample_data.csv'

diarization_llama_output_path = '/scratch/gpfs/jf3375/modern_family/output/Diarization_llama2/13b'

whisper_df = pd.read_csv(os.path.join(whisper_output_path, input_filename))

# Need to cut dataframe to ensure that the input text length does not exceed maximum tokens
# Since token is not equal to the number of words, would divide max_length by 2 to ensure the buffer
dataframe_segment_maxstrings = max_length-len(main_question_bgn) - len(examples)

# First identify if the split of dataframe is necessary
start_rowidx = 0
total_num_strings = 0
seg_num_strings = 0
sec_num_strings = 0
results_all_df = pd.DataFrame()

for current_rowidx, text in enumerate(list(whisper_df['text'])):
    total_num_strings += len(text)
    seg_num_strings += len(text)
    # Consider the condition in which the segment to last row does not exceed the maximum
    if seg_num_strings > dataframe_segment_maxstrings or current_rowidx == whisper_df.shape[0]-1:
        whisper_df_cut = whisper_df.iloc[start_rowidx:current_rowidx+1]
        # Restart the segment
        start_rowidx = current_rowidx+1
        seg_num_strings = 0

        # Pass the text inputs into the Llama2 model
        main_question = get_main_question(whisper_df_cut,
                                      main_question_bgn = main_question_bgn,
                                      examples = examples)

        full_prompt = get_full_prompt(systemprompt, main_question)
        print('full prompt')
        print(full_prompt)

        # Return the answer
        # Set Options to Get Deterministic Response
        sequences = pipeline(
            full_prompt,
            return_full_text=False,  # Not repeat the question
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            temperature=temperature  # Currently does not support the set of 0
        )
        print('----------------------------------------------------------------')
        print('Answer Started')

        results_segment_df = pd.DataFrame()
        segments = []
        speakers = []
        segmentids = []

        for seq in sequences:
            txt = seq['generated_text']
            print(f"Result: %{txt}")
            output = txt[txt.find("{"):txt.rfind("}")+1]
            data = json.loads(output)
            for segment in data['conversation']:
                segmentids.append(segment['segment id'])
                segments.append(segment['current segment'])
                speakers.append(segment['speaker changes'])

        results_segment_df['segment id'] = segmentids
        results_segment_df['text'] = segments
        results_segment_df['speaker_change_llama2'] = speakers
        results_all_df = pd.concat([results_all_df, results_segment_df])
results_all_df.to_csv(os.path.join(diarization_llama_output_path, input_filename), index = False)


