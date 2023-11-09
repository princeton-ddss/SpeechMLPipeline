import os

from Whisper import whisper_transcription

def main():
    # Sound File Input Path
    # soundfile_input_path = '/Users/jf3375/Desktop/DDSS/Projects/NJFS/audio_speech/data/njfs/output_cut/audio'
    soundfile_input_path = '/drives/Njfs-Audio/output/preprocess_data/output_cut/1007/audio'

    # Model path
    # model_path = '/Users/jf3375/PycharmProjects/TestMLEnvironment/models/Whisper/large-v2.pt'
    model_path = '/drives/Njfs-Audio/models/Whisper/large-v2.pt'

    # Output path
    output_path = '/drives/Njfs-Audio/output/whisper_result'

    os.chdir(soundfile_input_path)

    for input_filename in os.listdir(soundfile_input_path):
        if input_filename.endswith('.wav'):
            whisper_transcription(input_filename, model_path, output_path)




if __name__ == '__main__':
    main()