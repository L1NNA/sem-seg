from typing import Optional
from transformers import RobertaTokenizerFast, AutoTokenizer

GRAPH_CODE_BERT = 'microsoft/graphcodebert-base'
LONGFORMER = 'allenai/longformer-base-4096'
SENT_BERT = 'sentence-transformers/all-MiniLM-L12-v2'
TOKENIZER = {
    'tokenizer': None
}

def get_tokenizer(config=None) -> RobertaTokenizerFast:
    if TOKENIZER['tokenizer'] is None:
        model_path = get_model_path(None if config is None else config.bert_name)
        if model_path == GRAPH_CODE_BERT:
            TOKENIZER['tokenizer'] = RobertaTokenizerFast.from_pretrained(GRAPH_CODE_BERT)
        else:
            TOKENIZER['tokenizer'] = AutoTokenizer.from_pretrained(model_path)
    return TOKENIZER['tokenizer']

def get_model_path(bert_name) -> str:
    if bert_name == 'longformer':
        return LONGFORMER
    elif bert_name == 'sentbert':
        return SENT_BERT
    else:
        return GRAPH_CODE_BERT

