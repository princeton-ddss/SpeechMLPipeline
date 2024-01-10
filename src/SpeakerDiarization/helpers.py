from bisect import bisect_left

def merge_diarization_audio_results_with_whisper(timestamp_speaker, whisper_df):
    ''''
    Append speaker diarization results in Whisper results
    '''

    timestamp_list= list(timestamp_speaker.keys())

    speakers= [None] * whisper_df.shape[0]
    speaker_change = [None] * whisper_df.shape[0]

    prev_speaker, curr_speaker = None, None

    for idx, segment in whisper_df.iterrows():
        # account for lack of zero timestamp
        timestamp_idx = max(bisect_left(timestamp_list, segment['start']) - 1, 0)

        curr_speaker = timestamp_speaker[timestamp_list[timestamp_idx]]

        speakers[idx] = curr_speaker

        if curr_speaker != prev_speaker:
            speaker_change[idx] = True
        else:
            speaker_change[idx] = False

        prev_speaker = curr_speaker
    return speakers, speaker_change