import json
import pandas as pd

from huggingface_hub import snapshot_download

from LLaMA_HuggingFace_Offline import get_full_prompt
from LLaMA_HuggingFace_Offline import setup_LLaMA_model_tokenizer

from prompt1 import systemprompt, main_question_bgn, questions_answers_dict
from prompt_template import get_main_question

# Set llama model directory
llama_model_dir = "/scratch/gpfs/jf3375/models/huggingface/hub/llama"

# Find the name of downloaded folder llama where config.json is
llama_folder_name = 'models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8'

# Download LLaMA Model to Local Folder
download = False
if download:
    snapshot_download(repo_id = "meta-llama/Llama-2-7b-chat-hf",  cache_dir= llama_model_dir)

# Import LLaMA Tokenizer and Pipeline Offline
# One GPU: device = '0'
# Two GPUs: device ='0,1'
# Find the torch device 0 name
# torch.cuda.get_device_name(torch.device('cuda:0'))
# Check gpu resource
# nvidia-smi

tokenizer, pipeline = setup_LLaMA_model_tokenizer(llama_model_dir, llama_folder_name, gpu_num_devices = '0')

# Set parameters of the model to answer
# The input includes the question itself; Need to set up max_length long enough so all the answers would return
temperature = 0.1
top_p = 0
top_k = 1
max_length = 3000

input_file = '/scratch/gpfs/jf3375/llama_run/inputs/family_sample.csv'
# input_path = '/Users/jf3375/PycharmProjects/SpeechMLPipeline/output/Whisper'

main_question = get_main_question(input_file, main_question_bgn)

full_prompt = get_full_prompt(systemprompt, main_question, questions_answers_dict)

# Output CSV File
output_filename = '/scratch/gpfs/jf3375/llama_run/outputs/family_sample_LLaMA_output.csv'

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
print('prompt1')

results = pd.DataFrame()
sentences = []
speakers = []

for seq in sequences:
    txt = seq['generated_text']
    print(f"Result: %{txt}")
    output = txt[txt.find("{")-1:txt.rfind("}")+1]
    data = json.loads(output)
    for sentence in data['conversation']:
        sentences.append(sentence['sentence'])
        speakers.append(sentence['speaker'])

results['sentences'] = sentences
results['speakers'] = speakers
results['filename'] = input_file.split('/')[-1].split('.')[0]

results.to_csv(output_filename, index = False)


# Process Final Results and Output it to csv file
