from os import listdir
from os.path import isfile, join
from multiprocessing import Pool

import torch
from transformers import RobertaTokenizer

from data_loader.js_utils import generate_pairs, STRING_TOKEN, STRICT_TOKEN

tokenizer_name = 'microsoft/graphcodebert-base'
bert_tokenizer:RobertaTokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)


def generate_segments(js_path, max_size):

    tokens = []
    segment = 0

    prev_label = None

    for token, label in generate_pairs(js_path):
        for str_token in bert_tokenizer.tokenize(token):
            if str_token == STRICT_TOKEN:
                str_token = '@the'
            elif str_token == STRING_TOKEN:
                str_token = '@a'

            tokens.append(str_token)
            if prev_label is None:
                prev_label = label
            elif segment == 0 and prev_label != label:
                segment = 1

            if len(tokens) == max_size:
                token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
                yield token_ids, segment
                tokens.clear()
                segment = 0
                prev_label = None

    if len(tokens):
        token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        token_ids += [bert_tokenizer.pad_token_id] * (max_size - len(tokens))
        yield token_ids, segment


def save_tensors(src_dir, file_name:str, dest_dir, max_tokens):
    index = 0
    file_short = file_name[:file_name.rfind('.')]
    for token_ids, segment in generate_segments(join(src_dir, file_name), max_tokens):
        dest_name = join(dest_dir, f'{file_short}_{index}_{segment}.pt')
        x = torch.tensor(token_ids)
        torch.save(x, dest_name)
        index += 1

def process_js_files(src_dir, dest_dir, max_tokens, processes=12):
    js_files = [f for f in listdir(src_dir) if isfile(join(src_dir, f)) and f.endswith('.js')]
    js_files = [(src_dir, f, dest_dir, max_tokens) for f in js_files]

    with Pool(processes) as p:
        p.starmap(save_tensors, js_files)

