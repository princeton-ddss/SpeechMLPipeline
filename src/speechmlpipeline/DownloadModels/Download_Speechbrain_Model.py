import os

from speechbrain.pretrained import SpeakerRecognition

def download_speechbrain_model(download_model_path, model_folder= 'speechbrain'):
    verification_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                     savedir = os.path.join(download_model_path, model_folder))


