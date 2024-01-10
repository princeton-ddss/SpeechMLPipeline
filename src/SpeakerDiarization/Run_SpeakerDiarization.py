import os
import pandas as pd
import spacy

from speaker_diarization_main_function import run_speaker_diarization_models

hf_access_token = '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'
soundfile_input_path = '/Users/jf3375/Desktop/modern_family/sample_data'
pyannote_model_path = '/Users/jf3375/Dropbox (Princeton)/models/Pyannote/Diarization'
whisper_output_path = '/Users/jf3375/Desktop/modern_family/output/Whispertimestamped'
diarization_output_path = '/Users/jf3375/Desktop/modern_family/output/diarization'
diarization_llama2_output_path = '/Users/jf3375/Desktop/modern_family/output/diarization/Llama2'
#load nlp
nlp_model = spacy.load("en_core_web_lg")



input_filename = 'sample_data.WAV'
device = 'cpu'
min_speakers = 2
max_speakers = 10

csv_filename = input_filename.split('.')[0] + '.csv'
whisper_df = pd.read_csv(os.path.join(whisper_output_path, csv_filename))

diarization_models = ['pyannote', 'clustering', 'NLP', 'llama2']

whisper_diarization_df = run_speaker_diarization_models(soundfile_input_path, input_filename, device, min_speakers, max_speakers,
                                   whisper_df, diarization_models, pyannote_model_path, nlp_model, diarization_output_path,
                                   diarization_llama2_output_path,
                                   hf_access_token)







