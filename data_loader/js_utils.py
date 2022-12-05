from typing import List
import collections, json

from jsbeautifier.javascript.tokenizer import Tokenizer, TOKEN
from jsbeautifier.core.token import Token
from jsbeautifier.javascript.options import BeautifierOptions


STRING_TOKEN = '<strict>'
STRICT_TOKEN = '<string>'


# https://github.com/evmar/python-sourcemap/blob/master/smap.py
MappingState = collections.namedtuple(
    'MappingState',
    [
        'dst_line', 'dst_col',
        'src', 'src_line', 'src_col',
        'name'
    ]
)


# Mapping of base64 letter -> integer value.
B64 = dict(
    (c, i) for i, c in
    enumerate(
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        '0123456789+/'
    )
)


def parse_vlq(segment) -> List[int]:
    """Parse a string of VLQ-encoded data.

    Args:
        segment (str): the VLQ-encoded segmentation

    Raises:
        Exception: shift failed

    Returns:
        List[int]: a list of integers.
    """

    values = []

    cur, shift = 0, 0
    for c in segment:
        val = B64[c]
        # Each character is 6 bits:
        # 5 of value and the high bit is the continuation.
        val, cont = val & 0b11111, val >> 5
        cur += val << shift
        shift += 5

        if not cont:
            # The low bit of the unpacked value is the sign.
            cur, sign = cur >> 1, cur & 1
            if sign:
                cur = -cur
            values.append(cur)
            cur, shift = 0, 0

    if cur or shift:
        raise Exception('leftover cur/shift in vlq decode')

    return values


def parse_smap(f:str):

    smap = json.loads(f)
    sources = smap['sources']
    names = smap['names']
    mappings = smap['mappings']
    lines = mappings.split(';')

    dst_col, src_id, src_line, src_col, name_id = 0, 0, 0, 0, 0
    for dst_line, line in enumerate(lines):
        segments = line.split(',')
        dst_col = 0
        for segment in segments:
            if not segment:
                continue
            parse = parse_vlq(segment)
            dst_col += parse[0]

            src = None
            name = None
            if len(parse) > 1:
                src_id += parse[1]
                src = sources[src_id]
                src_line += parse[2]
                src_col += parse[3]

                if len(parse) > 4:
                    name_id += parse[4]
                    name = names[name_id]

            assert dst_line >= 0
            assert dst_col >= 0
            assert src_line >= 0
            assert src_col >= 0

            yield MappingState(dst_line, dst_col, src, src_line, src_col, name)


def generate_token(raw_string:str):
    tokenizer = Tokenizer(raw_string, BeautifierOptions())
    tokens:List[Token] = tokenizer.tokenize()

    # add tokens
    for token in tokens:
        if token.type == TOKEN.EOF:
            continue
        elif token.type == TOKEN.STRING:
            if token.text == "'use strict'" or '"use strict"':
                yield STRICT_TOKEN
            else:
                yield STRING_TOKEN
        else:
            yield token.text


def generate_pairs(js_path):

    # read JavaScript 
    mapping_path = js_path + '.map'
    with open(js_path, 'r', encoding='utf-8') as js_f, open(mapping_path, 'r', encoding='utf-8') as map_f:
        js_f.readline()
        raw_string = js_f.readline()
        raw_mapping = map_f.read()

    prev_src = None
    prev_col = 0

    # iterate through mapping states and extract tokens
    for state in parse_smap(raw_mapping):

        if prev_src == state.src:
            continue

        if prev_src:
            block = raw_string[prev_col:state.dst_col]
            for token in generate_token(block):
                yield token, prev_src

        prev_src = state.src
        prev_col = state.dst_col

    # read rest of the file
    if prev_src:
        block = raw_string[prev_col:]
        for token in generate_token(block):
            yield token, prev_src

