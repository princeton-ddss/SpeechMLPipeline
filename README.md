[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10712895.svg)](https://doi.org/10.5281/zenodo.10712895)

## SpeechMLPipeline #
[**SpeechMLPipeline**](https://github.com/princeton-ddss/SpeechMLPipeline) is a Python package for users to run the complete speech machine learning pipeline via one simple function
(Audio-to-Text Transcription, Speaker Change Detection, and Speaker Identification) to get 
transcriptions with speaker labels from input audio files. SpeechMLPipeline applys and implements the most widely used and the innovative machine learning
models at each step of the pipeline:
 * Audio-to-Text Transcription: [OpenAI Whisper with timestamp adjustment](https://github.com/linto-ai/whisper-timestamped)
 * Speaker Change Detection: [PyAnnotate](https://huggingface.co/pyannote/speaker-diarization-3.1), [Audio-based Spectral Clustering Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection), [Text-based Llama2-70b Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection), [Rule-based NLP Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection), [Ensemble Audio-and-text-based Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection)
 * Speaker Detection: [Speechbrain](https://github.com/speechbrain/speechbrain)

The [OpenAI Whisper](https://github.com/linto-ai/whisper-timestamped) is selected for the Transcription as it is the most accurate model available for English transcription. The OpenAI Whisper with timestamp adjustment is used to reduce the
misalignment between the timestamps and the transcription texts by identifying the silence parts and predicting timestamps at the word level.

The [PyAnnotate models](https://huggingface.co/pyannote/speaker-diarization-3.1) is by far one of the most popular models for speaker diarization. The speaker change detection results are directly inferred from speaker diarization results.

The [Audio-based Spectral Clustering Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is developed by extracting audio features from Librosa and applying spectral clustering to audio features. This model is one of the most common speaker change detection models used in academic research.

The [Text-based Llama2-70b Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is developed by asking Llama2 if the speaker changes across two consecutive text segments by understanding the interrelationships between these two texts via their semantic meaning. 

The [Rule-based NLP Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is applied to detect speaker change by analyzing text using well-defined rules developed by human comprehension. 

The [Ensemble Audio-and-text-based Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is built by ensembling audio-based or text-based speaker change detection models. The voting methods are used to aggregate the predictions of the speaker change detection models above except for Rule-based NLP model.
The aggregated predictions are then corrected by Rule-based NLP model.

The [Speechbrain models](https://github.com/speechbrain/speechbrain) are used to perform the speaker identification by comparing the similarities between the vector embeddings of each input audio segment and labelled speakers audio segments.

## Create New Python Environment to Avoid Packages Versions Conflict If Needed
```
python -m venv <envname>
source <envname>/bin/activate
```

## Install **speechmlpipeline** and its dependencies via Github
```
git lfs install
git clone https://github.com/princeton-ddss/SpeechMLPipeline
cd <.../SpeechMLPipeline>
pip install -r requirements.txt
pip install .
```
## Install **speechmlpipeline** via Pypi
```
pip install speechmlpipeline
```

## Download Models Offline to Run Them without Internet Connection
### Download PyAnnotate Models using Git Large File Storage (LFS)

PyAnnotate models are already in the **models** folder of the current repo. 

To download PyAnnotate models, please git clone the repo first.
```
git lfs install
git clone https://github.com/princeton-ddss/SpeechMLPipeline
```
To use the PyAnnotate models, please replace <local_path> with the local parent folder of the downloaded AudioAndTextBasedSpeakerChangeDetection repo in **models/pyannote3.1/Diarization/config.yaml** and
**models/pyannote3.1/Segmentation/config.yaml**.

### Download Spacy, Llama2, and Speechbrain Models by using the Download Module in the Repo
<hf_access_token> is the access token to Hugging Face.
Please create a [Hugging Face account](https://huggingface.co/) if it does not exist.  
The new access token could be created by following the [instructions](https://huggingface.co/docs/hub/en/security-tokens).

<models_list> is the list of names of models to be downloaded. Usually, the value of models_list should be set as 
['whisper', 'speechbrain', 'llama2-70b'].

<download_model_path> is the local path where all the downloaded models would be saved.

```python
from speechmlpipeline.DownloadModels.download_models_main_function import download_models_main_function

download_models_main_function(<download_model_path>, <models_list>, <hf_access_token>)
```




## Usage
The complete pipeline could be ran by using **run_speech_ml_pipeline** function.

Please view the function and its inputs description inside the Python file **src/speechmlpipeline/main_pipeline_local_function.py**.

Please view the sample codes to run the function in **sample_run.py** and **sample_run_existingllama2output.py** in the src/speechmlpipeline.
```python
from main_pipeline_local_function import TranscriptionInputs, SpeakerChangeDetectionInputs, 
    EnsembleDetectionInputs, SpeakerIdentificationInputs, run_speech_ml_pipeline

# Run Whole Pipeline except for Downloading Models
run_speech_ml_pipeline(transcription = <transcription_inputs>,
                       speakerchangedetection=<detection_inputs>, ensembledetection=<ensemble_detection_inputs>,
                       speakeridentification=<speaker_identification_inputs>)
```

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
