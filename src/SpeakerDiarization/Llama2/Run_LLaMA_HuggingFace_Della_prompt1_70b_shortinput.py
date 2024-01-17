
from Llama2_Diarizatioin_Main_Function import run_llama_diarization

# Set llama model directory
llama_model_local_dir = "/scratch/gpfs/jf3375/models/huggingface/hub/llama"
llama_model_size = '70b'


whisper_output_path = '/scratch/gpfs/jf3375/modern_family/output/Whispertimestamped'
input_filename = 'sample_data.csv'
diarization_llama_output_path = '/scratch/gpfs/jf3375/modern_family/output/Diarization_llama2/70b'

# Run Main Function
results_all_df = run_llama_diarization(whisper_output_path, input_filename, diarization_llama_output_path,
                           llama_model_local_dir, llama_model_size)

