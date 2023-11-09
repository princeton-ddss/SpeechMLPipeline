import pandas as pd
def get_main_question(input_file, main_question_bgn):
    text_df = pd.read_csv(input_file)
    sentences = list(text_df['text'])

    for idx, sentence in enumerate(sentences):
        sentence = '"sentence": "{}", "speaker":"" \n'.format(sentence)
        main_question_bgn += sentence

    main_question_bgn += ']\n}'
    return main_question_bgn