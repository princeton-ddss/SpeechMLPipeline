import os
from joblib import Parallel, delayed

from Whisper import whisper_transcription

def main():
    # Sound File Input Path
    soundfile_input_path = '/Users/jf3375/Desktop/DDSS/Projects/NJFS/audio_speech/data/njfs/output_cut/audio'

    # Model path
    model_path = '/Users/jf3375/PycharmProjects/TestMLEnvironment/models/Whisper/large-v2.pt'

    # Output path
    output_path = '/Users/jf3375/Desktop/DDSS/Projects/NJFS/audio_speech/data/njfs/whisper_result'

    os.chdir(soundfile_input_path)

    Parallel(n_jobs=-1)(delayed(whisper_transcription)(input_filename, model_path, output_path)
                        for input_filename in os.listdir(soundfile_input_path)
                                    if input_filename.endswith('.wav'))


if __name__ == '__main__':
    main()