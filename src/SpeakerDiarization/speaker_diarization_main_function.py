import pandas as pd
import os

from SpectralClustering.diarization_clustering import spectralclustering_speakerdiarization
from PyAnnote.diarization_pyannote import pyannote_speakerdiarization
from NLP.diarization_NLP import nlp_speakerdiarization
from helpers import merge_diarization_audio_results_with_whisper

def run_speaker_diarization_models(soundfile_input_path, input_filename, device, min_speakers, max_speakers,
                                   whisper_df, diarization_models, pyannote_model_path, nlp_model, diarization_output_path,
                                   diarization_llama2_output_path,
                                   hf_access_token):
    if 'clustering' in diarization_models:
        # Run diarization based on the clustering of sounds
        timestamp_speaker_clustering = spectralclustering_speakerdiarization(soundfile_input_path, input_filename, device,
                                                   min_speakers, max_speakers)

        print('finish clustering')
        speakers_clustering, speaker_change_clustering = merge_diarization_audio_results_with_whisper(timestamp_speaker_clustering, whisper_df)
        whisper_df['speaker_spectralclustering'] = speakers_clustering
        whisper_df['speaker_change_spectralclustering'] = speaker_change_clustering
    if 'pyannote' in diarization_models:
        timestamp_speaker_pyannote = pyannote_speakerdiarization(soundfile_input_path, input_filename,
                                                                 pyannote_model_path, hf_access_token,
                                                                 min_speakers, max_speakers)

        print('finish pyannote')
        speakers_pyannote, speaker_change_pyannote = merge_diarization_audio_results_with_whisper(timestamp_speaker_pyannote, whisper_df)
        whisper_df['speaker_pyannote'] = speakers_pyannote
        whisper_df['speaker_change_pyannote'] = speaker_change_pyannote
    if 'NLP' in diarization_models:
        whisper_df['speaker_change_NLP'] = nlp_speakerdiarization(whisper_df, nlp_model)
    if 'llama2' in diarization_models:
        # Should Run llama2 automatically
        input_csvfilename = input_filename.split('.')[0] + '.csv'
        df_llama2 = pd.read_csv(os.path.join(diarization_llama2_output_path, input_csvfilename))
        # drop duplicates segment id on llama output if the output has
        df_llama2 = df_llama2.drop_duplicates(subset=['segmentid'])
        # drop text column as the audio analysis dataframe has it
        df_llama2 = df_llama2.drop(columns='text')
        # merge two dataframe based on segment id
        whisper_df = pd.merge(whisper_df, df_llama2, on='segmentid', how='left')

    whisper_df.to_csv(os.path.join(diarization_output_path, input_csvfilename), index=False)
    return whisper_df
