[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10712894.svg)](https://doi.org/10.5281/zenodo.10712894)

## SpeechMLPipeline #
[**SpeechMLPipeline**](https://github.com/princeton-ddss/SpeechMLPipeline) is a Python package for users to run the complete speech machine learning pipeline via one simple function
to get transcriptions with speaker labels from input audio files. SpeechMLPipeline applys and implements the most widely used and the innovative machine learning
models at each step of the pipeline:
 * **Audio-to-Text Transcription**: [OpenAI Whisper with timestamp adjustment](https://github.com/linto-ai/whisper-timestamped)
 * **Speaker Change Detection**: [PyAnnotate](https://huggingface.co/pyannote/speaker-diarization-3.1), [Audio-based Spectral Clustering Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection), [Text-based Llama2-70b Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection), [Rule-based NLP Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection), [Ensemble Audio-and-text-based Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection)
 * **Speaker Identification**: [Speechbrain](https://github.com/speechbrain/speechbrain)

**Audio-to-text Transcription**
* The [OpenAI Whisper](https://github.com/linto-ai/whisper-timestamped) is selected for the audio-to-text transcription as it is the most accurate model available for English transcription. The OpenAI Whisper with timestamp adjustment is used to reduce the
misalignment between the timestamps and the transcription texts by identifying the silence parts and predicting timestamps at the word level.

**Speaker Change Detection**
* The [PyAnnotate models](https://huggingface.co/pyannote/speaker-diarization-3.1) is by far one of the most popular models for speaker diarization. It detects speaker change by applying clustering methods based on audio features. The speaker change detection results are directly inferred from speaker diarization results.

* The [Audio-based Spectral Clustering Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is developed by extracting audio features from Librosa and applying spectral clustering to audio features. This model is one of the most common speaker change detection models used in academic research.

* The [Text-based Llama2-70b Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is an innovative speaker change detection model based on LLMs. It is developed by asking Llama2 if the speaker changes across two consecutive text segments by understanding the interrelationships between these two texts via their semantic meaning. 

* The [Rule-based NLP Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is applied to detect speaker change by analyzing text using well-defined rules developed by human comprehension. 

* The [Ensemble Audio-and-text-based Speaker Change Detection Model](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection) is built by ensembling audio-based or text-based speaker change detection models. The voting methods are used to aggregate the predictions of the speaker change detection models above except for Rule-based NLP model.
The aggregated predictions are then corrected by Rule-based NLP model.

**Speaker Identification**
* The [Speechbrain models](https://github.com/speechbrain/speechbrain) are used to perform the speaker identification by comparing the similarities between the vector embeddings of each input audio segment and labelled speakers audio segments.

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
### Download Spacy NLP Model by Running Commands below in Terminal
```
python -m spacy download en_core_web_lg
```

### Download Whisper, Llama2, and Speechbrain Models by using the Download Module in the Repo
<hf_access_token> is the access token to Hugging Face.
Please create a [Hugging Face account](https://huggingface.co/) if it does not exist. The new access token could be created by following the [instructions](https://huggingface.co/docs/hub/en/security-tokens).

<models_list> is the list of names of models to be downloaded. Usually, the value of models_list should be set as 
['whisper', 'llama2-70b', 'speechbrain'].

<download_model_path> is the local path where all the downloaded models would be saved.

```python
from speechmlpipeline.DownloadModels.download_models_main_function import download_models_main_function

download_models_main_function(<download_model_path>, <models_list>, <hf_access_token>)
```

### Download PyAnnotate Models using Dropbox Link

To download PyAnnotate models, please download pyannote3.1 folder in this [Dropbox Link](https://www.dropbox.com/scl/fo/tp2uryaq81sze2l0yuxb9/ACgXWOr7Be1ZZovz7xNSuTs?rlkey=9c2z50pjbjhoo3vz4dbxlmlcf&st=fukejg4l&dl=0).

To use the PyAnnotate models, please replace <local_path> with the local parent folder of the downloaded pyannote3.1 folder in **pyannote3.1/Diarization/config.yaml** and
**pyannote3.1/Segmentation/config.yaml**.


## Usage
The complete pipeline could be run by using **run_speech_ml_pipeline** function which could be directly imported as 
```Python
from speechmlpipeline import run_speech_ml_pipeline
```

The **run_speech_ml_pipeline** function takes four classes instances corresponding to each step in the Speech Machine Learning Pipeline as the inputs:

* **transcription**:
TranscriptionInputs Class to specify inputs to run OpenAI Whisper for Audio-to-Text Transcription with Timestamps Adjustment
* **speakerchangedetection**:
SpeakerChangeDetectionInputs Class to specify inputs to run various models including PyAnnote Model, Spectral Clustering, Llama2, and NLP Rule-Based Analysis for Speaker Change Detection
* **ensembledetection**:
EnsembleDetectionInputs Class to specify inputs to build an Ensemble Model of Speaker Change Detection by considering both audio and textual features
* **speakeridentification**:
SpeakerIdentificationInputs Class to specify inputs to run Speechbrain Verification Model for Speaker Identification

To run the complete pipeline, the function could be called as
```Python
run_speech_ml_pipeline(transcription = TranscriptionInputs(...),
                       speakerchangedetection=SpeakerChangeDetectionInputs(...), ensembledetection=EnsembleDetectionInputs(...),
                       speakeridentification=SpeakerIdentificationInputs(...))
```

To run any particular steps, please simply just use the inputs corresponding to the particular steps.
For instance, to run all steps of the pipeline with the existing transcriptions:
```Python
run_speech_ml_pipeline(speakerchangedetection=SpeakerChangeDetectionInputs(...), ensembledetection=EnsembleDetectionInputs(...),
                       speakeridentification=SpeakerIdentificationInputs(...))
```

For instance, to run speaker change detection with the existing transcriptions:
```Python
run_speech_ml_pipeline(speakerchangedetection=SpeakerChangeDetectionInputs(...), ensembledetection=EnsembleDetectionInputs(...))
```

For instance, to run speaker identification with the existing transcriptions and speaker change detection results:
```Python
run_speech_ml_pipeline(speakeridentification=SpeakerIdentificationInputs(...))
```

Please view the descriptions below to specify the attributes of the class instance corresponding to each step of the pipeline.
* **TranscriptionInputs** 
    * audio_file_input_path: A path which contains the audio file
    * audio_file_input_name: A audio file name containing the file type
    * whisper_model_path: A path where the Whisper model files are saved
    * whisper_output_path: A path to save the csv file of transcription outputs
    * device: Torch device type to run the model; If device is set as None, GPU would be automatically used if it is available. 
    * only_run_in_english: True or False to Indicate if Whisper would only be run when
    the identified langauge in the audio file is English
* **SpeakerChangeDetectionInputs**
    * audio_file_input_path: A path which contains an input audio file
    * audio_file_input_name: A audio file name containing the file type
    * min_speakers: The minimal number of speakers in the input audio file
    * max_speakers: The maximal number of speakers in the input audio file
    * whisper_output_path: A path where a Whisper transcription output csv file is saved
    * whisper_output_file_name: A Whisper transcription output csv file name ending with .csv
    * detection_models: A list of names of speaker change detection models to be run
    * detection_output_path: A path to save the speaker change detection output in csv file
    * hf_access_token: Access token to HuggingFace
    * llama2_model_path: A path where the Llama2 model files are saved
    * pyannote_model_path: A path where the Pyannote model files are saved
    * device: Torch device type to run the model; If device is set as None, GPU would be automatically used if it is available.
    * detection_llama2_output_path: A path where the pre-run Llama2 speaker change detection output in csv file
    * temp_output_path: A path to save the current run of Llama2 speaker change detection output
    to avoid future rerunning
* **EnsembleDetectionInputs**
    * detection_file_input_path: A path where the speaker change detection output in csv file is saved
    * detection_file_input_name: A speaker change detection output csv file name ending with .csv
    * ensemble_output_path: A path to save the ensemble detection output in csv file
    * ensemble_voting: A list of voting methods to be used to build the final ensemble model
* **SpeakerIdentificationInputs**
    * detection_file_input_path: A path where the speaker change detection output in csv file is saved
    * detection_file_input_name: A speaker change detection output csv file name ending with .csv
    * audio_speaker_file_input_path:  A path which contains a verified audio file of each speaker
    * audio_file_input_path: A path which contains an input audio file
    * verification_model_path: A path where the speaker verification model files are saved, default to None
    * speaker_change_col: A column name in the detection output csv file which specifies which speaker
    change detection model result is used for speaker identification
    * verification_score_threshold: A score threshold in which the speaker would be identified as "OTHERS"
    if the verification score is below this threshold，ranging from negative value to 1
    * identification_output_path: A path to save the speaker identification output in csv file
    * temp_output_path: A path to save the temporary cut audio file of each segment

Please view the sample codes to run the function in **sample_run.py** and **sample_run_existingllama2output.py** in the **src/speechmlpipeline** folder. 
For detailed functions and class decriptions, please refer to **src/speechmlpipeline/main_pipeline_local_function.py**

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
