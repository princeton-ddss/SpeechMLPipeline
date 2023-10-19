'''
Functions to Preprocess Video/Audio for ML Models Inputs
'''

#!/usr/bin/env python
# coding: utf-8

import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips

# Create the error class of not equal length of video and its related audio
class VideoAudioLengthNotEqual(Exception):
    "The Audio of Video Does Not Have Same Length as Video"
    pass

# Run the ffmpeg to remove background noise
def remove_noise(input_filename, input_path, output_path):
    command = ["ffmpeg", "-hide_banner -nostats", "-i", "{}/{}".format(input_path, input_filename),
               "-af",  "'afftdn=nf=-25,afftdn=nf=-25,highpass=f=200,lowpass=f=3000'",
              "{}/{}".format(output_path, input_filename)]
    subprocess.run(" ".join(command), shell=True)

# Run the ffmpeg to find silence parts of audio and video
# thresholds=-60, 60 db is normal conversation level
# duration=0.5, Remove silence of more than 0.5 second: avoid removing the natural silence between words and speaker changes
def output_silence_timestamps(input_filename, input_path, output_path, threshold=-60, duration=1):
    filename = input_filename.split(".")[0]
    # -vn: only keeps audio stream
    command = ["ffmpeg", "-vn", "-hide_banner -nostats",
                               "-i", "{}/{}".format(input_path, input_filename),
                               "-af", "'silencedetect=n={}dB:d={}'".format(threshold, duration),
                               "-f null - 2>&1 | grep 'silence_end' | awk '{{print $5 $7 $8}}' > {}/{}.txt".format(output_path, filename)]
    print(" ".join(command))
    subprocess.run(" ".join(command), shell=True)

# Cut video and audio based on silence timestamps
def cut_video_and_audio_based_on_silence(input_filename_noft, video_filetype, audio_filetype,
                                         input_silence_filename, input_video_path, input_silencets_path, output_path):
    # Ease in duration between cuts
    ease = 0.0
    minimum_duration = 1.0

    # number of clips generated
    count = 0
    # start of next clip
    last = 0

    input_file = "{}/{}.{}".format(input_video_path, input_filename_noft, video_filetype)
    output_audio_file = "{}/audio/{}.{}".format(output_path, input_filename_noft, audio_filetype)
    output_video_file = "{}/video/{}.{}".format(output_path, input_filename_noft, video_filetype)

    in_handle = open('{}/{}'.format(input_silencets_path, input_silence_filename), "r", errors='replace')
    video = VideoFileClip(input_file)
    audio = AudioFileClip(input_file)
    if video.duration != audio.duration:
        raise VideoAudioLengthNotEqual
    full_duration = video.duration

    clips = []
    audios = []
    while True:
        line = in_handle.readline()

        if not line:
            break

        end, duration = line.strip().split('silence_duration:')

        to = float(end) - float(duration)

        start = float(last)
        clip_duration = float(to) - start

        if clip_duration < minimum_duration:
            continue

        if full_duration - to < minimum_duration:
            continue

        if start > ease:
            start -= ease

        #     print("Clip {} (Start: {}, End: {})".format(count, start, to))
        video_clip = video.subclip(start, to)
        audio_clip = audio.subclip(start, to)

        clips.append(video_clip)
        audios.append(audio_clip)
        last = end
        count += 1

    if full_duration - float(last) > minimum_duration:
        clips.append(video.subclip(float(last) - ease))
        audios.append(audio.subclip(float(last) - ease))

    processed_video = concatenate_videoclips(clips)
    processed_audio = concatenate_audioclips(audios)
    processed_video.audio = processed_audio

    processed_video.write_videofile(
        output_video_file,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    processed_audio.write_audiofile(
        output_audio_file
    )

    in_handle.close()
    if processed_video.duration != processed_audio.duration:
        raise VideoAudioLengthNotEqual
    video.close()
    audio.close()
    processed_video.close()
    processed_audio.close()
