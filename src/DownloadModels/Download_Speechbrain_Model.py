from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from Download_Llama_Model import download_llama_model

embedding_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                 savedir=embedding_model_path)
verification_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                     savedir=verification_model_path)

