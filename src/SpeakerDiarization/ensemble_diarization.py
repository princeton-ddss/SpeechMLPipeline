import os
import pandas as pd

from helpers import map_empty_string_to_true, use_value_major

def ensemble_diarization(diarization_file_path, diarization_file_name):
    diarization_csv = pd.read_csv(os.path.join(diarization_file_path, diarization_file_name))

    # Find all speaker change columns of all models
    speaker_change_columns = [colname for colname in diarization_csv.columns if colname.startswith('speaker_change')]
    speaker_change_nonlp_columns = [colname for colname in speaker_change_columns if 'NLP' not in colname]

    # Fill Missing Values of llama2 by TRUE to reduce false positive rate
    # Astype bool would automatically convert missing values/empty string to FALSE
    if 'speaker_change_llama2' in speaker_change_columns:
        # Fill Missing Values
        diarization_csv['speaker_change_llama2'] = diarization_csv['speaker_change_llama2'].apply(map_empty_string_to_true)
        # Convert string True or False to Bool Type
        diarization_csv['speaker_change_llama2'] = diarization_csv['speaker_change_llama2'].astype(bool)

    # Ensemble diarization results
    # If the ensemble model of all models agrees that the speakerchanges = FALSE, would return ensemble as false and merge sounds
    # e.g.: False or False -> False; True or False -> True
    # Reduce the false positive rate to identify speakerchanges=TURE by avoding merging sounds of different poeple together
    diarization_csv['speaker_change_ensemble'] = diarization_csv[speaker_change_nonlp_columns].any(axis = 1)

    # Correct Results based on Rule-Based analysis
    # If the NLP determines the speaker changes, use NLP result to identify speaker change
    if 'speaker_change_NLP' in speaker_change_columns:
        diarization_csv['speaker_change_ensemble'] = diarization_csv.apply(lambda df: use_value_major(value_major= df['speaker_change_NLP'],
                                                                                                      value_minor =df['speaker_change_ensemble']), axis = 1)
        # Convert string/object type in NLP to  bool type; Otherwise, 'False' would become True
        diarization_csv['speaker_change_ensemble'] = diarization_csv['speaker_change_ensemble'].astype(bool)
    return diarization_csv

# Diarization Files Info
diarization_file_path = '/Users/jf3375/Desktop/modern_family/output/diarization'
diarization_file_name ='sample_data.csv'

ensemble_csv = ensemble_diarization(diarization_file_path, diarization_file_name)