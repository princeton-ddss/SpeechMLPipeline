import os
from pyannote.audio import Pipeline



def pyannote_speakerdiarization(soundfile_input_path, input_filename,
                                 pyannote_model_path, hf_access_token,
                                           min_speakers, max_speakers):

    if os.path.exists(pyannote_model_path):
        diarization_pipeline = Pipeline.from_pretrained(os.path.join(pyannote_model_path, 'config.yaml'))
    else:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_access_token)


    diarization_result = diarization_pipeline(os.path.join(soundfile_input_path, input_filename),
                                                min_speakers = min_speakers, max_speakers = max_speakers)

    timestamp_speaker = {}
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        timestamp_speaker[turn.start] = speaker

    return timestamp_speaker




