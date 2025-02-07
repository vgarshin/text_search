#!/usr/bin/env python
# coding: utf-8

import torch
import transformers
import Levenshtein
from searchtools import wordsearchutils
from tqdm.auto import tqdm


def pool(hidden_state, mask, pooling_method='cls'):
    if pooling_method == 'mean':
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == 'cls':
        return hidden_state[:, 0]


def text_chunck(idx, text, inputs, num_tokens=512, overlap_tokens=4):
    token_pos_start = min(
        len(inputs['input_ids']) - 2, 
        max(
            1, 
            idx * (num_tokens - overlap_tokens) + 1
        )
    )
    token_pos_end = min(
        len(inputs['input_ids']) - 2, 
        token_pos_start + num_tokens - 1
    )
    start = inputs.token_to_chars(token_pos_start).start
    end = inputs.token_to_chars(token_pos_end).end
    return text[start:end], start, end


def generate_text_chunks(text, inputs, num_tokens=512, overlap_tokens=4):
    idxs = len(inputs['input_ids']) // (num_tokens - overlap_tokens)
    for idx in range(idxs):
        yield text_chunck(idx=idx, text=text, 
                          inputs=inputs, 
                          num_tokens=num_tokens, 
                          overlap_tokens=overlap_tokens)


def ner_text(text, model, tokenizer, ner_entities, th,
             num_tokens=512, overlap_tokens=4):
    inputs = tokenizer(text)
    idxs = len(inputs['input_ids']) // (num_tokens - overlap_tokens)
    results = []
    nlp = transformers.pipeline(
        'ner', 
        model=model, 
        tokenizer=tokenizer, 
        aggregation_strategy='simple'
    )
    for chunk, start, end in tqdm(generate_text_chunks(
        text, inputs, num_tokens=num_tokens, 
        overlap_tokens=overlap_tokens
    ), desc='NER search', total=idxs):
        ner_results = nlp(chunk)
        

        def glue(i, ners, symb='#'):
            nrw, nre = ners[i]['word'], ners[i]['end']
            if i == len(ners) - 1: return nrw, nre
            if symb in ners[i + 1]['word']:
                next_nr = glue(i + 1, ners, symb=symb)
                return nrw + next_nr[0].replace(symb, ''), next_nr[1]
            else:
                return nrw, nre

        
        for i, nr in enumerate(ner_results):
            if (nr['entity_group'] in ner_entities) \
            and (nr['score'] >= th) \
            and ('#' not in nr['word']):
                nr['word'], nr['end'] = glue(i, ner_results)
                nr['start'] += start
                nr['end'] += start
                results.append(nr)
    return results


def ners_embed(ners, model, tokenizer, pool_layer=False, max_len=64):
    sentences = [x['word'] for x in ners]
    encoded_input = tokenizer(
        sentences, 
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
    for i, ner in enumerate(ners):
        ner['embeddings'] = embeddings[i]
    return ners


def ners_match(ners_wanted, ners, th, text='', text_lim=100):
    matches_found = []
    for ner in ners:
        embs_ner = ner['embeddings']
        for ner_wtd in ners_wanted:
            embs_ner_wtd = ner_wtd['embeddings']
            match_score = float(embs_ner @ embs_ner_wtd.T)
            match_score_lv = Levenshtein.ratio(ner['word'], ner_wtd['word'])
            match_score_lcs = wordsearchutils.lcs_ratio(ner['word'], ner_wtd['word'])
            if match_score >= th:
                match_found = {
                    'entity_group': ner_wtd[ 'entity_group'],
                    'word': ner_wtd['word']
                }
                match_found['match_start'] = ner['start']
                match_found['match_end'] = ner['end']
                match_found['match_word'] = ner['word']
                match_found['match_score'] = match_score
                match_found['match_score_lv'] = match_score_lv
                match_found['match_score_lcs'] = match_score_lcs
                if text:
                    match_found['text'] = text[ner['start'] - text_lim : ner['start']].replace('\n', ' ') + ' >>>>> '
                    match_found['text'] += text[ner['start'] : ner['end']].replace('\n', ' ')
                    match_found['text'] += ' <<<<< ' + text[ner['end'] : ner['end'] + text_lim].replace('\n', ' ')
                    match_found['where_in_text'] = str(round(100 * ner['start'] / len(text), 1)) + '%'
                matches_found.append(match_found)
    return matches_found


def tokens_embed(encoded_input, model, pool_layer=False):
    with torch.no_grad():
        output = model(**encoded_input)
    if pool_layer:
        embeddings = pool(
            output.last_hidden_state, 
            encoded_input['attention_mask'],
            pooling_method='cls'  # or 'mean'
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    else:
        embeddings = output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


def embs_match(embs_wanted, embs):
    match_score = float(embs_wanted @ embs.T)
    return match_score


def search_text(text_wanted, text, model, pool_layer,
                tokenizer, th, shift_tokens=.5):
    inputs_wanted = tokenizer(
        text_wanted, 
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    embs_wanted = tokens_embed(inputs_wanted, model, pool_layer)
    inputs = tokenizer(text)
    num_tokens = len(inputs_wanted['input_ids'][0])
    shift_tokens = max(1, int(shift_tokens * len(inputs_wanted['input_ids'][0])))
    overlap_tokens = num_tokens - shift_tokens
    idxs = len(inputs['input_ids']) // (num_tokens - overlap_tokens)
    results = []
    for chunk, start, end in tqdm(generate_text_chunks(
        text, inputs, num_tokens=num_tokens, 
        overlap_tokens=overlap_tokens
    ), desc='text search', total=idxs):
        result = {}
        inputs_chunk = tokenizer(
            chunk, 
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        embs = tokens_embed(inputs_chunk, model, pool_layer)
        match_score = embs_match(embs_wanted, embs)
        if match_score > th:
            result['match_start'] = start
            result['match_end'] = end
            result['match_text'] = text[start:end].replace('\n', ' ')
            result['match_score'] = match_score
            result['where_in_text'] = str(round(100 * start / len(text), 1)) + '%'
            results.append(result)
    return results