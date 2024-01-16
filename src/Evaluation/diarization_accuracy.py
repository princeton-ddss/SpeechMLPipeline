import os
import pandas as pd

# define the function to calculate accuracy score by treating the missing values as errors
def accuracy_score_considering_missing(predicted_not_missing_values, predicted_not_missing_labelled_values, num_total_values):
    num_accurate_values = sum(predicted_not_missing_values == predicted_not_missing_labelled_values)
    accuracy_score = num_accurate_values/num_total_values
    return accuracy_score


input_path = '/Users/jf3375/Desktop/modern_family/output/diarization/Llama2'
filename = 'sample_data_modelsizescomparison.csv'
input_df = pd.read_csv(os.path.join(input_path, filename))

# Get total number of values which we are supposed to do inference
num_total_values = input_df.shape[0]

# Convert string values to bool type
input_df['speaker_change_true'] = input_df['speaker_change_true'].astype(bool)

# Delete missing values
# Should only apply bool after deleting missing values as the bool function would automatically convert empty strings/missing values to false
input_df_7b = input_df[['speaker_change_true', '7b']].dropna().astype(bool)
input_df_13b = input_df[['speaker_change_true', '13b']].dropna().astype(bool)
input_df_70b = input_df[['speaker_change_true', '70b']].dropna().astype(bool)

accuracy_scores = {'llama_70b': accuracy_score_considering_missing(input_df_70b['speaker_change_true'], input_df_70b['70b'], num_total_values),
                   'llama_13b': accuracy_score_considering_missing(input_df_13b['speaker_change_true'], input_df_13b['13b'], num_total_values),
                   'llama_7b': accuracy_score_considering_missing(input_df_7b['speaker_change_true'], input_df_7b['7b'], num_total_values)}

# num of missing values
missing_values = {'llama_70b': sum(input_df['70b'].isna())/input_df.shape[0],
                   'llama_13b':  sum(input_df['13b'].isna())/input_df.shape[0],
                   'llama_7b': sum(input_df['7b'].isna())/input_df.shape[0]}