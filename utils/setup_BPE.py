from transformers import RobertaTokenizer, RobertaTokenizerFast

GRAPH_CODE_BERT = 'microsoft/graphcodebert-base'
TOKENIZER = {
    'tokenizer': None
}

def get_tokenizer() -> RobertaTokenizerFast:
    if TOKENIZER['tokenizer'] is None:
        TOKENIZER['tokenizer'] = RobertaTokenizerFast.from_pretrained(GRAPH_CODE_BERT)
    return TOKENIZER['tokenizer']
