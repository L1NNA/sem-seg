from transformers import RobertaTokenizer


STRICT_TOKEN = '<STRICT>'
SEP_TOKEN = '<SEP>'

tokenizer_name = 'microsoft/graphcodebert-base'
bert_tokenizer:RobertaTokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)

bert_tokenizer.add_special_tokens({'additional_special_tokens': [STRICT_TOKEN], 
                                   'sep_token': SEP_TOKEN
                                   })

bert_tokenizer.save_pretrained('./cache')