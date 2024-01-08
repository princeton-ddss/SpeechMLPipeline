import os

from Whispertimestamped import whisper_transcription

def main():
    # Sound File Input Path
    soundfile_input_path = '/scratch/gpfs/jf3375/modern_family/audio/sample_data'
    # soundfile_input_path = '/Users/jf3375/Desktop/modern_family/video/cut'

    # Model path
    model_path = '/scratch/gpfs/jf3375/models/Whisper/large-v2.pt'
    # model_path = '/Users/jf3375/Dropbox (Princeton)/models/Whisper/large-v2.pt'

    # Output path
    output_path = '/scratch/gpfs/jf3375/modern_family/output/Whispertimestamped'
    # output_path = '/Users/jf3375/Desktop/modern_family/output/Whispertimestamped'

    os.chdir(soundfile_input_path)

    for input_filename in sorted(os.listdir(soundfile_input_path)):
        if input_filename.endswith('.wav') or input_filename.endswith('.WAV'):
            whisper_transcription(input_filename, model_path, output_path)




if __name__ == '__main__':
    main()