#!/usr/bin/env python
# coding: utf-8

import os
import glob
import torch
import json
import argparse
import datetime
import transformers
from tqdm.auto import tqdm
from textpreproc import file2text
from searchtools import nerutils, wordsearchutils

TS = datetime.datetime.now()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Searches named entities with NER models in the texts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('configpath', help='path to config file')
    parser.add_argument('workpath', help='path to working directory')
    parser.add_argument(
        '--resultpath', 
        type=str, 
        default='matches',
        help='directory to save results'
    )
    parser.add_argument(
        '--wordspath', 
        type=str,
        default='wanted_words',
        help='directory to read txt files with words to search'
    )
    parser.add_argument(
        '--thw', 
        type=float, 
        default=.7,
        help='threshold for words match in range from 0 to 1'
    )
    parser.add_argument(
        '--nerpath', 
        type=str, 
        default='wanted',
        help='directory to read files with NERs to search'
    )
    parser.add_argument(
        '--th',
        type=float, 
        default=.8,
        help='threshold for NERs search in range from 0 to 1'
    )
    parser.add_argument(
        '--thm',
        type=float, 
        default=.8,
        help='threshold for NERs match in range from 0 to 1'
    )
    parser.add_argument(
        '--litpath', 
        type=str, 
        default='',
        help='directory to files with arbitary phrases to search'
    )
    parser.add_argument(
        '--thmt',
        type=float, 
        default=.6,
        help='threshold for phrases match in range from 0 to 1'
    )
    parser.add_argument(
        '--shift',
        type=float, 
        default=.5,
        help='window shift for search (used for phrases search, from 0 to 1)'
    )
    args = parser.parse_args()

    WORK_PATH = args.workpath
    CONFIG_FILE = args.configpath  # 'config.json'
    RESULT_PATH = args.resultpath
    
    WANTED_WORDS_PATH = args.wordspath
    THW = args.thw
    
    WANTED_PATH = args.nerpath
    TH = args.th
    THM = args.thm
    
    WANTED_LIT_PATH = args.litpath  # or 'wanted_lit'
    THMT = args.thmt
    TXT_SHIFT = args.shift
    
    try:
        with open(CONFIG_FILE, 'r') as file:
            config_data = json.load(file)
    except FileNotFoundError as e:
        print(e)
    
    MODEL_NER = config_data['model_ner']
    MODEL_EMB = config_data['model_emb']
    EMB_LEN = config_data['emb_len']
    NUM_TOKENS = config_data['num_tokens']
    OVERLAP = config_data['overlap']
    
    tokenizer_ner = transformers.AutoTokenizer.from_pretrained(MODEL_NER)
    model_ner = transformers.AutoModelForTokenClassification.from_pretrained(MODEL_NER)
    print('NER model and tokenizer loaded:', MODEL_NER)
    tokenizer_emb = transformers.AutoTokenizer.from_pretrained(MODEL_EMB)
    model_emb = transformers.AutoModel.from_pretrained(MODEL_EMB)
    print('embedding model and tokenizer loaded:', MODEL_EMB)
    
    wanted_word_files = glob.glob(f'{WORK_PATH}/{WANTED_WORDS_PATH}/*')
    wanted_words = ''
    for file_path in wanted_word_files:
        add_text =  file2text.file_text(file_path)
        wanted_words += '\n' + add_text
    print('total words found:', len(wanted_words.split('\n')))
    
    wanted_files = glob.glob(f'{WORK_PATH}/{WANTED_PATH}/*')
    text = ''
    for file_path in wanted_files:
        add_text =  file2text.file_text(file_path)
        if 'Список' in file_path:
            add_text = add_text.replace(',', '')
        text += '\n' + add_text
    
    ners_wanted = nerutils.ner_text(
        text=text,
        model=model_ner,
        tokenizer=tokenizer_ner,
        ner_entities=['ORG', 'PER'],
        th=TH,
        num_tokens=NUM_TOKENS,
        overlap_tokens=OVERLAP
    )
    ners_wanted = [x for x in ners_wanted if len(x['word'].split()) > 1]
    print('total NERs found:', len(ners_wanted))
    
    ner_file_path = f'{WORK_PATH}/{RESULT_PATH}/ners_wanted_{str(TS).replace(' ', '_')}.json'
    with open(ner_file_path, 'w', encoding='utf-8') as file:
        json.dump(eval(str(ners_wanted)), file, indent=4, ensure_ascii=False)
    print('NERs saved to:', ner_file_path)
    
    ners_wanted = nerutils.ners_embed(
        ners_wanted, 
        model=model_emb,
        tokenizer=tokenizer_emb,
        pool_layer=False,
        max_len=EMB_LEN
    )
    
    if WANTED_LIT_PATH:

        
        def first_part(line, samples):
            res = ''
            for s in samples:
                res += line.split(s)[0] if s in line else ''
            return res
        
        
        samples = [
            ', решение',
            '. Решение',
            ' (решение'
        ]
        wanted_lit_files = glob.glob(f'{WORK_PATH}/{WANTED_LIT_PATH}/*')
        wanted_items = []
        for file_path in wanted_lit_files:
            with open(file_path, 'r', encoding='cp1251') as file:
                for line in file:
                    line = line.split(';')[1].replace('\n', '').strip()
                    if len(line.split()) > 1:
                        wanted_items.append(first_part(line, samples))
    
    target_files = glob.glob(f'{WORK_PATH}/target/*/*')
    print('total files:', len(target_files))
    tps = set([x.split('.')[-1] for x in target_files])
    print('all types:', tps)
    total = 0
    for tp in list(tps):
        count = len([x for x in target_files if x.endswith(tp)])
        total += count
        print(f'{tp} files:', count)
    assert total == len(target_files)
    
    log_file_path = f'{WORK_PATH}/{RESULT_PATH}/matches_found_{str(TS).replace(' ', '_')}.json'
    all_matches_found = []
    for file_path in tqdm(target_files, desc='files'):
        ners_data = {}
        matches_found = []
        item_results = []
    
        # read files to text
        text = file2text.file_text(file_path)
        if not text:
            continue
    
        # search words in text
        matches_found.extend(
            wordsearchutils.find_words(
                wanted_words, 
                text,
                model_emb,
                tokenizer_emb,
                th=THW
            )
        )
    
        # search for NER in text
        ners_data['path'] = file_path
        ners_data['ners'] = nerutils.ner_text(
            text=text, 
            model=model_ner, 
            tokenizer=tokenizer_ner,
            ner_entities=['ORG', 'PER'],
            th=TH,
            num_tokens=NUM_TOKENS, 
            overlap_tokens=OVERLAP
        )
    
        # search for arbitary text
        if WANTED_LIT_PATH:
            wanted_items = wanted_items[:2]
            for item in tqdm(wanted_items, desc='custom search'):
                text_found = nerutils.search_text(
                    text_wanted=item, 
                    text=text,
                    model=model_emb,
                    pool_layer=False,
                    tokenizer=tokenizer_emb, 
                    th=THMT,
                    shift_tokens=TXT_SHIFT
                )
                if text_found:
                    item_results.append(
                        {
                            'text': item.replace('\n', ' ').strip(),
                            'text_found': text_found
                        }
                    )
    
        # match found entities with target list
        if ners_data['ners']:
            ners = nerutils.ners_embed(
                ners_data['ners'], 
                model=model_emb,
                tokenizer=tokenizer_emb,
                pool_layer=False,
                max_len=EMB_LEN
            )
            matches_found.extend(
                nerutils.ners_match(
                    ners_wanted, 
                    ners, 
                    th=THM, 
                    text=text, 
                    text_lim=500
                )
            )
    
        # collect results and write to a file
        if len(matches_found) + len(item_results) > 0:
            all_matches_found.append(
                    {
                    'path': ners_data['path'],
                    'matches_found': matches_found,
                    'items_found': item_results
                }
            )
            with open(log_file_path, 'w', encoding='utf-8') as file:
                json.dump(all_matches_found, file, indent=4, ensure_ascii=False)
    
    # save final result as excel output
    file2text.results_xlsx(
        all_matches_found, 
        log_file_path, 
        key='path', 
        sub_key=['matches_found']
    )

    print('finished')
