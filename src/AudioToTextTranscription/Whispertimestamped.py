'''
Functions to Apply Whisper with Improved TimeStamped Accuracy for Audio-Text Transcription and Output Results in CSV to Create Training Data
'''

#!/usr/bin/env python
# coding: utf-8
import os
import whisper_timestamped as whisper
import pandas as pd

def whisper_transcription(soundfile_input_path, model_path, output_path):
    if '/' in soundfile_input_path:
        soundfile_name = soundfile_input_path.split("/")[-1].split(".")[0]
    else:
        soundfile_name = soundfile_input_path.split(".")[0]


    audio = whisper.load_audio(soundfile_input_path)
    whisper_model = whisper.load_model(model_path) # Could not use predownloaded Whisper model for accurate timestamp

    # Translate text into Whisper
    result = whisper.transcribe(whisper_model, audio, beam_size=5, best_of=5,
                                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                vad = 'auditok')
    segment_len = len(result['segments'])
    transcribe_df = pd.DataFrame()
    start_list = []
    end_list = []
    text_list = []

    # Process Whisper Outputs
    # Should concat duplicated continuous text columns together
    first_segment = result['segments'][0]
    start_idx, end_idx, prev_text = first_segment['start'], first_segment['end'], first_segment['text']

    for idx, segment in enumerate(result['segments'][1:]):
        if prev_text != segment['text']:
            # output previous same rows into df
            start_list.append(start_idx)
            end_list.append(end_idx)
            text_list.append(prev_text)
            start_idx, end_idx, prev_text = segment['start'], segment['end'], segment['text']
        else:
            end_idx = segment['end']

    # Append last segment to df
    start_list.append(start_idx)
    end_list.append(end_idx)
    text_list.append(prev_text)

    transcribe_df['start'], transcribe_df['end'], transcribe_df['text'] = start_list, end_list, text_list
    transcribe_df['file_name'] = soundfile_name
    transcribe_df['speaker'] = ''

    # Remove leading and trailing whitespaces from whisper outputs
    transcribe_df['text'] = transcribe_df['text'].apply(lambda x: x.strip())
    # Create unique id of whisper segment for later merge
    transcribe_df['segmentid'] = list(range(transcribe_df.shape[0]))

    transcribe_df.to_csv(os.path.join(output_path, '{}.csv'.format(soundfile_name)), index=False)
    return transcribe_df

