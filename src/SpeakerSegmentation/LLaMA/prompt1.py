import pandas as pd
# Set up prompt
# Five typical examples are enough for few shot learning
# ~ 4000 examples are needed for the fine-tuning

systemprompt = """You are the expert of speaker diarization. 
You are familiar with the daily life conversation between different family members.
You are very capable of identifying how children and parents speak differently.
You are very proficient in identifying which family member speaks which sentence.
"""

# Question_Answer Example to ensure correct format
question1 = """"Could you identify which family member spoke which sentence in the following conversations? Please put the answer in the speaker section of json format below.

{"conversation":[
{"sentence": "What is the breakfast today?", "speaker":""},
{"sentence": "Omelete with ham and cheese", "speaker":""},
{"sentence": "That sounds delicious.", "speaker":""},
{"sentence": "Could you eat faster? We need to hurry to go to school.", "speaker":""}
]
}

{"conversation":[
{"sentence": "Put your shoes on.", "speaker":""},
{"sentence": "Did you know that we don’t actually vote for president, we vote for people who vote for president?", "speaker":""},
{"sentence": " I don’t care, put your shoes on.", "speaker":""},
{"sentence": "Why do we have shoes anyway?", "speaker":""},
{"sentence": "Just put your shoes on.", "speaker":""}
]
}
"""

answer1 = """
{"conversation":[
{"sentence": "What is the breakfast today?", "speaker":"kids"},
{"sentence": "Omelete with ham and cheese", "speaker":"parents"},
{"sentence": "That sounds delicious.", "speaker":"kids"},
{"sentence": "Could you eat faster? We need to hurry to go to school.", "speaker":"parents"}
]
}

{"conversation":[
{"sentence": "Put your shoes on.", "speaker":"parents"},
{"sentence": "Did you know that we don’t actually vote for president, we vote for people who vote for president?", "speaker":"kids"},
{"sentence": " I don’t care, put your shoes on.", "speaker":"parents"},
{"sentence": "Why do we have shoes anyway?", "speaker":"kids"},
{"sentence": "Just put your shoes on.", "speaker":"parents"}
]
}
"""

main_question = """"
The answer is great! Could you identify which family member spoke which sentence in the following conversation? 
Please answer exactly only in the json format below.
Your answer should only answer the question once. 
Please do not include any additional information before and after your answers.
Please think step by step.
Please make sure you identify and fill all the speakers of each sentence.

{"conversation":[
{"sentence": "Who is Jerry?", "speaker":""},
{"sentence": "The character in a story", "speaker":""},
{"sentence": "Oh Ok. What he is doing?", "speaker":""},
{"sentence": "Could you go to sleep? You are sick today", "speaker":""},
{"sentence": "Okay. Okay", "speaker":""}
]
}
"""

# Formulate main question by using csv file
main_question_bgn = """
The answer is great! Could you identify which family member spoke which sentence in the following conversation? 
Please answer exactly only in the json format below.
Your answer should only answer the question once. 
Please do not include any additional information before and after your answers.
Please think step by step.
Please make sure you identify and fill all the speakers of each sentence.

{"conversation":[
"""

questions_answers_dict = {question1: answer1}
