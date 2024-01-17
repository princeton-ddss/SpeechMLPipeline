import os
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from pydub import AudioSegment

from src.SpeakerDiarization.ensemble_diarization import ensemble_diarization

# Set up inputs foro speaker identification
embedding_model_path = '/Users/jf3375/Dropbox (Princeton)/models/Speechbrain/EncoderClassifier'
verification_model_path = '/Users/jf3375/Dropbox (Princeton)/models/Speechbrain/SpeakerRecognition'
sound_speaker_path = '/Users/jf3375/Desktop/modern_family/characters_sounds'
identification_results_path = '/Users/jf3375/Desktop/modern_family/output/identification'
temp_folder_path = '/Users/jf3375/Desktop/modern_family/temp'
audio_path = '/Users/jf3375/Desktop/modern_family/sample_data'
filename_noftype = 'sample_data'
audio_name = '{}.WAV'.format(filename_noftype)

# List speakers and their related sounds files
sound_speaker_files = [filename for filename in sorted(os.listdir(sound_speaker_path)) if filename.endswith('WAV')]
speakers = [filename.split('.')[0] for filename in sound_speaker_files]

# Ensemble diarization results
diarization_file_path = '/Users/jf3375/Desktop/modern_family/output/diarization'
diarization_file_name ='{}.csv'.format(filename_noftype)
diarization_ensemble_csv = ensemble_diarization(diarization_file_path, diarization_file_name)

# Set Up Speaker Change Columns to Merge Sounds
speaker_change_column = 'speaker_change_ensemble'

# Would indetify verification result only if the score is above 0
verification_score_threshold = 0

# Import speaker identification models
if os.path.exists(embedding_model_path):
    embedding_model = EncoderClassifier.from_hparams(source=embedding_model_path)
else:
    embedding_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=embedding_model_path)

if os.path.exists(verification_model_path):
    verification_model = SpeakerRecognition.from_hparams(source = verification_model_path)
else:
    verification_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=verification_model_path)

# Import full_audio
full_audio = AudioSegment.from_wav(os.path.join(audio_path, audio_name))


# Speaker Identification for each audio segment
speaker_identify_allsegments = []
start_segment = diarization_ensemble_csv.iloc[0]['start']
end_segment = diarization_ensemble_csv.iloc[0]['end']
start_row_idx = 0
end_row_idx = 0

# Would ignore the first sentence
for current_row_idx, row in diarization_ensemble_csv.iterrows():
    if current_row_idx == 0: # No speaker changes information of the first row
        continue

    # Deal with the last row condition
    # If speaker_change = True, merge previous false speaker change segments together
    if row[speaker_change_column]:
        audio_segment = full_audio[start_segment * 1e3:end_segment * 1e3]
        audio_segment.export(os.path.join(temp_folder_path, 'audio_segment_temp.wav'), format="wav")

        speaker_scores_segment = {}
        for idx, sound_speaker_file in enumerate(sound_speaker_files):
            score, _ = verification_model.verify_files(os.path.join(sound_speaker_path, sound_speaker_file),
                                                       os.path.join(temp_folder_path, 'audio_segment_temp.wav'))
            speaker_scores_segment[speakers[idx]] = score.item()

        # Find the optimal speaker
        speaker_identify  = max(speaker_scores_segment, key = speaker_scores_segment.get)
        print(speaker_scores_segment)
        print(speaker_identify)
        print(start_row_idx, end_row_idx)

        # Would identify speaker as OTHERS if the score is below the threshold such as 0
        if speaker_scores_segment[speaker_identify] < verification_score_threshold:
            speaker_identify = 'OTHERS'

        speaker_identify_allsegments.extend([speaker_identify]*(end_row_idx-start_row_idx+1))
        start_segment = row['start']
        end_segment = row['end']
        start_row_idx = current_row_idx
        end_row_idx = current_row_idx
    else:
        end_row_idx = current_row_idx
        end_segment = row['end']


speaker_identify_allsegments.append('random_last_row')
diarization_ensemble_csv['diarization_ensemble'] = speaker_identify_allsegments

# Remove previous append last row
diarization_ensemble_csv.iloc[:-1].to_csv(os.path.join(identification_results_path, diarization_file_name), index = False)