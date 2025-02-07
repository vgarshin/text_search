#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import ssl
import json
import argparse
import socket
from random import randint, uniform, choice, getrandbits
from urllib.request import (
    Request, 
    urlopen, 
    URLError, 
    HTTPError, 
    ProxyHandler, 
    build_opener, 
    install_opener)
from urllib.parse import quote, unquote
from bs4 import BeautifulSoup
from time import sleep, gmtime, strftime
from tqdm.auto import tqdm


class TextsCollector():
    def __init__(self, user_agent, proxies, timeout, max_counts, 
                 min_time_sleep, max_time_sleep, browser=False):
        self.user_agent = user_agent
        self.proxies = proxies
        self.timeout = timeout
        self.max_counts = max_counts
        self.min_time_sleep = min_time_sleep
        self.max_time_sleep = max_time_sleep
    
    def parce_content(self, url_page, 
                      proxies=None, file_content=False, json_data=None):
        counts = 0
        content = None
        while counts < self.max_counts:
            try:
                request = Request(url_page)
                request.add_header('User-Agent', self.user_agent)
                if proxies:
                    prx = choice(proxies)
                    proxy_support = ProxyHandler(prx)
                    opener = build_opener(proxy_support)
                    install_opener(opener)
                    response = urlopen(
                        request, 
                        context=ssl._create_unverified_context(),
                        timeout=self.timeout
                    )
                else:
                    if json_data:
                        response = urlopen(
                            request, 
                            data=json.dumps(json_data).encode('utf-8'),
                            timeout=self.timeout
                        )
                    else:
                        response = urlopen(request, timeout=self.timeout)
                if file_content:
                    content = response.read()
                else:
                    try:
                        content = response.read().decode(
                            response.headers.get_content_charset()
                        )
                    except:
                        content = None
                break
            except Exception as e:
                counts += 1
                print('ERROR request | ', url_page, ' | ', e, ' | counts: ', counts)
                sleep(uniform(
                    counts * self.min_time_sleep, counts * self.max_time_sleep
                ))
        return content
    
    def data_collect(self, url_page):
        # read number of pages
        content = self.parce_content(url_page=url_page)
        soup = BeautifulSoup(content, 'html.parser')
        total_num = soup.find('div', attrs={'class': 'summary'}).find_all('b')
        total_num = [x.text for x in total_num]
        num_page = int(total_num[0].split('-')[-1])
        total_num = int(total_num[-1])
        pages = total_num // num_page + 1
        data_table = []

        # iterate over pages to collect metadata for all texts
        for page in tqdm(range(pages), desc='pages'):
            content = self.parce_content(url_page=url_diplomas + f'&page={page + 1}')
            soup = BeautifulSoup(content, 'html.parser')
            page_table = soup.find_all('tr', attrs={'class': 'cursor-pointer'})
            for line in page_table:
                dict_line = {}
                data_line = line.find_all('td')
                dict_line['idx'] = line['data-key']
                dict_line['full_name'] = data_line[0].text
                dict_line['field'] = data_line[1].text
                dict_line['topic'] = data_line[2].text
                dict_line['supervisor'] = data_line[3].text
                dict_line['status'] = data_line[4].text
                dict_line['year'] = data_line[5].text
                dict_line['url'] = data_line[6].a['href']
                data_table.append(dict_line)
        return data_table
    
    def files_collect(self, base_url, save_path, data_table):
        log_file_path = f'{save_path}/data_log.json'
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as file:
                data_log = json.load(file)
        else:
            data_log = []
        idxs_loaded = [x['idx'] for x in data_log]
        for row in tqdm(data_table, desc='files'):
            if row['idx'] in idxs_loaded:
                continue
            try:
                row_url = base_url + row['url']
                content = self.parce_content(url_page=row_url)
                soup = BeautifulSoup(content, 'html.parser')
                data_row = soup.find(
                    'table', 
                    attrs={'class': 'table table-striped table-bordered detail-view'}
                ).find_all('tr')
                row['qualify'] = data_row[2].td.text
                row['url_handle'] = data_row[3].td.a['href']
                row['url_dspace'] = data_row[3].td.a['href'].replace(
                    'http://hdl.handle.net',
                    'https://dspace.spbu.ru/handle'
                )
                content = self.parce_content(url_page=row['url_dspace'])
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table', attrs={'class': 'table panel-body'})
                
                # create dirs and save files
                os.makedirs(f'{save_path}/{row["idx"]}', exist_ok=True)
                for td in table.find_all('td', attrs={'headers': 't1'}):
                    file_url = 'https://dspace.spbu.ru' + td.a['href']
                    file_name = f'{save_path}/{row["idx"]}/' + file_url.split('/')[-1]
                    content = self.parce_content(url_page=file_url, file_content=True)
                    with open(file_name, 'wb') as file:
                        file.write(content)
    
                # write logs as file
                data_log.append(row)
                with open(log_file_path, 'w') as file:
                    json.dump(data_log, file)
            except Exception as e:
                print('ERROR parce | ', row)
        return data_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collects diplomas as text files from SPBU internet site',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('savepath', help='path to save downloaded files')
    parser.add_argument(
        '--dpid', 
        type=int, 
        default=12,
        help='department id of papers published'
    )
    parser.add_argument(
        '--status', 
        type=int, 
        default=1,
        help='status id of papers published'
    )
    parser.add_argument(
        '--year', 
        type=int, 
        default=2023,
        help='year of papers published'
    )
    parser.add_argument(
        '--mintimesleep', 
        type=float, 
        default=.1,
        help='mininal time to sleep between downloads'
    )
    parser.add_argument(
        '--maxtimesleep',
         type=float, 
        default=2,
        help='mininal time to sleep between downloads'
    )
    args = parser.parse_args()
    
    USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 YaBrowser/19.6.1.153 Yowser/2.5 Safari/537.36'
    MIN_TIME_SLEEP = args.mintimesleep
    MAX_TIME_SLEEP = args.maxtimesleep
    MAX_COUNTS = 10
    TIMEOUT = 10
    WORK_PATH = args.savepath #'/home/jovyan/__CUSTOM/texts'
    
    search_params = {
        'name_ru': '',
        'title_ru': '',
        'editor_ru': '',
        'dp_id': args.dpid,
        'status': args.status,
        'year': args.year
    }
    search_url = '&'.join([
        f'GpSearch%5B{k}%5D={v}' for k, v in search_params.items()
    ])
    url_diplomas = f'https://diploma.spbu.ru/gp/index?{search_url}'
    collector = TextsCollector(
        user_agent=USER_AGENT, 
        proxies=None,
        timeout=TIMEOUT, 
        max_counts=MAX_COUNTS,
        min_time_sleep=MIN_TIME_SLEEP, 
        max_time_sleep=MAX_TIME_SLEEP
    )
    data_table = collector.data_collect(url_page=url_diplomas)
    data_log = collector.files_collect(
        base_url='https://diploma.spbu.ru', 
        save_path=WORK_PATH, 
        data_table=data_table
    )
    print('finished')
