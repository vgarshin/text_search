#!/usr/bin/env python
# coding: utf-8

import json
import pypdf
import zipfile
import pandas as pd
import xml.etree.ElementTree as ET
from striprtf import striprtf
from subprocess import Popen, PIPE

WP = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def file_text(file_path):
    text = ''
    if file_path.endswith(('docx', 'DOCX')):
        tree = doc_tree(file_path)
        text = doc_text(tree)
        text_tables = doc_texts_by_tables(tree)
    elif file_path.endswith(('doc', 'DOC')):
        text = olddoc_text(file_path)
    elif file_path.endswith(('rtf', 'RTF')):
        text = rtf_text(file_path)
    elif file_path.endswith(('pdf', 'PDF')):
        text = pdf_text(file_path)
    elif file_path.endswith(('json', 'JSON', 'ipynb')):
        text = json_text(file_path)
    elif file_path.endswith(('txt', 'TXT')):
        text = txt_text(file_path)
    elif file_path.endswith(('csv', 'CSV')):
        text = csv_text(file_path)
    elif file_path.endswith(('xlsx', 'XLSX')):
        text = xlsx_text(file_path)
    else:
        print(f'File `{file_path}` skipped.')
    return text


def olddoc_text(file_path):
    cmd = ['antiword', file_path]
    p = Popen(cmd, stdout=PIPE)
    stdout, stderr = p.communicate()
    text = stdout.decode('ascii', 'ignore').replace('|', '')
    return text


def doc_tree(file_path):
    doc = zipfile.ZipFile(file_path)
    xml_content = doc.read('word/document.xml')
    tree = ET.fromstring(xml_content)
    return tree


def doc_texts_by_tables(tree):
    body = tree.find(WP + 'body')
    tables = body.findall(WP + 'tbl')
    tables_texts = []
    for it, table in enumerate(tables):
        table_items = []
        for p in table.iter(WP + 'p'):
            p_text = [t.text for t in p.iter(WP + 't')]
            table_items.append(''.join(p_text))
        tables_texts.append(
            {it: ' '.join(table_items)}
        )
    return tables_texts


def doc_text(tree):
    paragraphs = []
    for paragraph in tree.iter(WP + 'p'):
        texts = [
            node.text 
            for node in paragraph.iter(WP + 't') 
            if node.text
        ]
        if texts:
            paragraphs.append(''.join(texts))
    return '\n\n'.join(paragraphs)


def pdf_text(file_path):
    assert file_path.endswith(('pdf', 'PDF'))
    reader = pypdf.PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text


def rtf_text(file_path):
    assert file_path.endswith(('rtf', 'RTF'))
    with open(file_path) as file:
        content = file.read()
        text = striprtf.rtf_to_text(content)
    return text


def json_text(file_path):
    assert file_path.endswith(('json', 'JSON', 'ipynb'))
    with open(file_path, 'r') as file:
        data = json.load(file)
    return json.dumps(data, ensure_ascii=False)


def txt_text(file_path):
    assert file_path.endswith(('txt', 'TXT'))
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def csv_text(file_path):
    assert file_path.endswith(('csv', 'CSV'))
    text = ''
    with open(file_path, 'r') as file:
        for line in file:
            text += line
    return text


def xlsx_text(file_path):
    assert file_path.endswith(('xlsx', 'XLSX'))
    df = pd.read_excel(file_path)
    return df.to_string()


def results_xlsx(data, file_path, key=None, sub_key=None):
    df = pd.json_normalize(data, sub_key, key)
    file_path = file_path.replace('.json', '.xlsx')
    df.to_excel(file_path)
    return file_path