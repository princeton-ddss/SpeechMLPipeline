'''
Functions to Apply Whisper for Audio-Text Transcription and Output Results in CSV to Create Training Data
'''

#!/usr/bin/env python
# coding: utf-8
import os
import whisper
import pandas as pd

def whisper_transcription(soundfile_input_path, model_path, output_path):
    if '/' in soundfile_input_path:
        soundfile_name = soundfile_input_path.split("/")[-1].split(".")[0]
    else:
        soundfile_name = soundfile_input_path.split(".")[0]

    whisper_model = whisper.load_model(name = model_path)

    # Translate text into Whisper
    result = whisper_model.transcribe(soundfile_input_path)
    segment_len = len(result['segments'])
    transcribe_df = pd.DataFrame()
    start_list = [0] * segment_len
    end_list = [0] * segment_len
    text_list = [0] * segment_len

    for idx, segment in enumerate(result['segments']):
        start_list[idx], end_list[idx], text_list[idx] = segment['start'], segment['end'], segment['text']

    transcribe_df['start'], transcribe_df['end'], transcribe_df['text'] = start_list, end_list, text_list
    transcribe_df['file_name'] = soundfile_name
    transcribe_df['speaker'] = ''

    transcribe_df.to_csv(os.path.join(output_path, '{}.csv'.format(soundfile_name)), index=False)
    return transcribe_df

