import os
import pandas as pd
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from pydub import AudioSegment

embedding_model_path = '/Users/jf3375/Dropbox (Princeton)/models/Speechbrain/EncoderClassifier'
verification_model_path = '/Users/jf3375/Dropbox (Princeton)/models/Speechbrain/SpeakerRecognition'
sound_speaker_path = '/Users/jf3375/Desktop/modern_family/characters_sounds'
whisper_results_path = '/Users/jf3375/Desktop/modern_family/output/Whispertimestamped'
speechbrain_results_path = '/Users/jf3375/Desktop/modern_family/output/SpeechBrain'
temp_folder_path = '/Users/jf3375/Desktop/modern_family/temp'
audio_path = '/Users/jf3375/Desktop/modern_family/audio'
audio_name = 'S01E01MODERNFAMILY.WAV'
audio_name_noftype = audio_name.split('.')[0]

sound_speaker_files = [filename for filename in sorted(os.listdir(sound_speaker_path)) if filename.endswith('WAV')]
speakers = [filename.split('.')[0] for filename in sound_speaker_files]
# Would trust verification result only if the score is above 0
verification_score_threshold = 0

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

# Remove beginning silence in audio: Whisper removes automatically

# Import Whisper Result
whisper_csv = pd.read_csv(os.path.join(whisper_results_path, audio_name_noftype + '.csv'))

# Speaker Diarization for each audio segment
final_diarization = {}
speaker_identify_allsegments = [None] * whisper_csv.shape[0]

for row_idx, row in whisper_csv.iterrows():
    print(row['text'])
    audio_segment = full_audio[row['start']*1e3:row['end']*1e3]
    audio_segment.export(os.path.join(temp_folder_path, 'audio_segment_temp_{}.wav'.format(row_idx)), format="wav")
    speaker_scores_segment = {}
    for idx, sound_speaker_file in enumerate(sound_speaker_files):
        score, _ = verification_model.verify_files(os.path.join(sound_speaker_path, sound_speaker_file),
                                                   os.path.join(temp_folder_path, 'audio_segment_temp_{}.wav'.format(row_idx)))
        speaker_scores_segment[speakers[idx]] = score.item()

    # Find the optimal speaker
    speaker_identify  = max(speaker_scores_segment, key = speaker_scores_segment.get)
    print(speaker_scores_segment)
    print(speaker_identify)

    # Would identify speaker as OTHERS if the score is below the threshold such as 70
    if speaker_scores_segment[speaker_identify] < verification_score_threshold:
        speaker_identify = 'OTHERS'

    speaker_identify_allsegments[row_idx] = speaker_identify

whisper_csv['speaker_speechbrain'] = speaker_identify_allsegments
whisper_csv.to_csv(os.path.join(speechbrain_results_path, '{}.csv'.format(audio_name_noftype)), index = False)





