# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

from llama import Llama, Dialog



def run_LLaMA_from_repo(
    systemprompt,
    main_question,
    output_path,
    output_filename,
    questions_answers_dict = None,
    ckpt_dir = str,
    tokenizer_path = str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        systemprompt (str): System Prompt
        main_question (str): Main Question for LLaMA to answer
        output_path (str): The directory to output LLaMA answwer
        output_file (str): The file to output LLaMA answer
        questions_answers_dict (dict): {Question1: Answer1, ..., QuestionN: AnswerN} for the few-shot learning
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    if not questions_answers_dict:
        prompt = [{"role": "system", "content": "{}".format(systemprompt)},
                  {"role": "user", "content": "{}".format(main_question)}]
    else:
        prompt = [0] * (2+2*len(questions_answers_dict))

        # Set System Prompt
        prompt[0] = {"role": "system", "content": "{}".format(systemprompt)}

        # Set Main Question
        prompt[-1] = {"role": "user", "content": "{}".format(main_question)}

        # Iterate question and answer to set few-shot learning
        idx = 1
        for question, answer in questions_answers_dict.items():
            prompt[idx] = {"role": "user", "content": "{}".format(question)}
            prompt[idx+1] = {"role": "assistant", "content": "{}".format(answer)}
            idx += 2

    print('prompt\n', prompt)
    dialogs:  List[Dialog] = [prompt]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")

        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        with open("{}/{}".format(output_path, output_filename), "w") as file:
            for line in result['generation']['content'].splitlines(True):
                file.write(line)
        print("\n==================================\n")



