from pydub import AudioSegment
from pydub.playback import play

# Functions Play Wav File
wavfile_input = '/Users/jf3375/Desktop/DDSS/Projects/NJFS/audio_speech/data/njfs/output_cut/audio/sample.wav'
audio = AudioSegment.from_wav(wavfile_input)
play(audio)