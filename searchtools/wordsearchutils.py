#!/usr/bin/env python
# coding: utf-8

import torch
import transformers
import Levenshtein


def lcs_ratio(X, Y):
    """
    Calculates Longest Common Subsequence ratio
    as `len(LCS) / max(len1, len2))`.

    Parameters
    ----------
    s1, s2 : Strings to compare.
    
    Returns
    -------
    ratio : float in range [0, 1]
        
    """
    m = len(X)
    n = len(Y)
    L = [[None]*(n + 1) for i in range(m + 1)] 
    for i in range(m + 1):
        for j in range(n + 1): 
            if i == 0 or j == 0: 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
    return L[m][n] / max(m, n)


def find_word(i, inputs):
    token = inputs.tokens()[i]
    word_idx = inputs.token_to_word(i)
    start = inputs.word_to_chars(word_idx).start
    end = inputs.word_to_chars(word_idx).end
    return token, start,  end


def match2words(words, model, tokenizer, max_len=64, pool_layer=False):
    if tokenizer:
        encoded_input = tokenizer(
            words, 
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
    with torch.no_grad():
        output = model(**encoded_input)
    if pool_layer:
        embeddings = pool(
            output.last_hidden_state, 
            encoded_input['attention_mask'],
            pooling_method='cls' # or 'mean'
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    else:
        embeddings = output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
    return float(embeddings[0] @ embeddings[1].T)


def find_words(wanted_words, text, model, tokenizer, th=.7, text_lim=100):
    wanted_words = wanted_words.lower()
    text = text.lower()
    wanted_inputs = tokenizer(
        wanted_words, 
        padding=True,
        truncation=True,
        return_attention_mask=False
    )
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_attention_mask=False
    )
    results = []
    for idxw, token_id in enumerate(wanted_inputs['input_ids']):
        if wanted_inputs.token_to_chars(idxw):
            token, start, end = find_word(idxw, wanted_inputs)
            wanted_word = wanted_words[start:end]
            if '#' not in token:
                idxs = [i for i, x in enumerate(inputs['input_ids']) if x == token_id]
                if idxs:
                    for idx in idxs:
                        token, start, end = find_word(idx, inputs)
                        word = text[start:end]
                        match_score = match2words(
                            [wanted_word, word], 
                            model,
                            tokenizer
                        )
                        if match_score > th:
                            res = {}
                            res['entity_group'] = 'WORD'
                            res['word'] = wanted_word
                            res['match_start'] = start
                            res['match_end'] = end
                            res['match_word'] = word
                            res['match_score'] = match_score
                            res['match_score_lv'] = Levenshtein.ratio(wanted_word, word)
                            res['match_score_lcs'] = lcs_ratio(wanted_word, word)
                            res['text'] = text[max(0, start - text_lim):start].replace('\n', ' ') + ' >>>>> '
                            res['text'] += text[start:end]
                            res['text'] += ' <<<<< ' + text[end:min(len(text), end + text_lim)].replace('\n', ' ')
                            res['where_in_text'] = str(round(100 * start / len(text), 1)) + '%'
                            results.append(res)
    return results