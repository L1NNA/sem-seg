from transformers import RobertaTokenizer

GRAPH_CODE_BERT = 'microsoft/graphcodebert-base'
TOKENIZER = {
    'tokenizer': None
}

def get_tokenizer() -> RobertaTokenizer:
    if TOKENIZER['tokenizer'] is None:
        TOKENIZER['tokenizer'] = RobertaTokenizer.from_pretrained(GRAPH_CODE_BERT)
    return TOKENIZER['tokenizer']

