from typing import Optional
from transformers import RobertaTokenizerFast, AutoTokenizer

GRAPH_CODE_BERT = 'microsoft/graphcodebert-base'
LONGFORMER = 'allenai/longformer-base-4096'
SENT_BERT = 'sentence-transformers/all-MiniLM-L12-v2'
TOKENIZER = {
    'tokenizer': None
}

def get_tokenizer(bert_name=None) -> RobertaTokenizerFast:
    if TOKENIZER['tokenizer'] is None:
        model_path = get_model_path(bert_name)
        if model_path == GRAPH_CODE_BERT:
            TOKENIZER['tokenizer'] = RobertaTokenizerFast.from_pretrained(GRAPH_CODE_BERT)
        else:
            TOKENIZER['tokenizer'] = AutoTokenizer.from_pretrained(model_path)
    return TOKENIZER['tokenizer']

def get_model_path(bert_name:Optional[str]=None) -> str:
    if bert_name is None:
        return GRAPH_CODE_BERT
    elif bert_name == 'graphcodebert':
        return GRAPH_CODE_BERT
    elif bert_name == 'longformer':
        return LONGFORMER
    elif bert_name == 'sentbert':
        return SENT_BERT
    else:
        raise Exception('AuotBert Model ' + bert_name + ' is not support')

