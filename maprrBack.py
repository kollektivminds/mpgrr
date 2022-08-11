#!/usr/bin/env python
# coding: utf-8

import os 
import time
import logging
import random
import pickle
import pandas as pd 
import re
import requests 
from bs4 import BeautifulSoup
from natasha import (
    Segmenter, 
    MorphVocab, 
    NewsEmbedding, 
    NewsMorphTagger, 
    NewsSyntaxParser, 
    NewsNERTagger, 
    PER, 
    NamesExtractor, 
    Doc)
from razdel import tokenize

logging.basicConfig(filename='maprr_out.log', encoding='utf-8', format='%(asctime)s %(message)s', level=logging.INFO)

lib_cols = ['title_ru', 'genre', 'text', 'title_en', '1st_line', 'author', 'comp_date', 'comp_loc', 'pub_src', '1st_pub', 'pub_year', 'pub_loc']
a_cols = ['name', 'birth', 'death', 'a_type', 'sex', 'occs', 'fam_soc_str', 'lit_affil', 'pol_affil', 'corp_type', 'corp_affil']

domain = 'https://maprr.iath.virginia.edu/'
tables = {
    'agents/': 323, 
    'works/': 648, 
    'place_based_concepts/': 316, 
    'locations/': 370, 
    'multivalent_markers/': 439
}

allURLs = [domain+item[0]+str(i) for item in tables.items() for i in range(1, item[1]+1)]

urls_to_visit = [] 
for t, i in list(tables.items())[:2]: 
    for j in range(0,i+1): 
        urls_to_visit.append(domain+t+str(j))

class maprr: 
    
    def __init__(self): 
        self.Wsoup = {} 
        self.Asoup = {}
        self.Ws = {}
        self.As = {}
    
    def get_htmlA(self): 
        """This function uses the list of Agent IDs from the tables dict above
        and the grabs it using requests before putting the html reponse in Asoup"""
        
        # initialize list of pages that don't return 200
        aberrantAs = []
        # go through list of Agents from 1 to the number defined in tables
        for i in range(1, (list(tables.values())[0]+1)):
        #for i in range(1, 11):
            # make url
            url = domain+list(tables.keys())[0]+str(i) 
            # initialize connection to url 
            with requests.get(url, verify=False) as r: 
                # log status code 
                logging.info(f"A{i} status code: {r.status_code}")
                # if connection is successful
                if r.status_code == 200: 
                    # make soup from html
                    s = BeautifulSoup(r.content, 'html.parser') 
                    # add to list of A soups
                    self.Asoup.update({i:s})
                else: 
                    # if connection is not successful, add to list
                    aberrantAs.append((i, r.status_code))
                    pass
                # wait a hot second or the server gets >:( 
                time.sleep(.1)
        # report list of A errors if there is one (there always is)
        if len(aberrantAs) > 0: 
            print(f"Aberrant agent pages are #s {aberrantAs}")
        else: 
            print(f"There were no aberrant agent pages!")

    def get_htmlW(self): 
        """This function uses the list of Work IDs from the tables dict above
        and the grabs it using requests before putting the html reponse in Wsoup"""
        
        # initialize list of pages that don't return 200
        aberrantWs = []
        # go through list of Words from 1 to the number defined in tables
        for i in range(1, (list(tables.values())[1]+1)):
        #for i in range(1, 11):
            # make url
            url = domain+list(tables.keys())[1]+str(i)
            # initialize connection to url 
            with requests.get(url, verify=False) as r: 
                # log status code
                logging.info(f"W{i} status code: {r.status_code}")
                # if connection is successful
                if r.status_code == 200: 
                    # make soup from html
                    s = BeautifulSoup(r.content, 'html.parser')
                    # add to list of A soups
                    self.Wsoup.update({i:s})
                else: 
                    # if connection is not successful, add to list
                    aberrantWs.append((i, r.status_code))
                    pass
                # wait a hot second or the server gets >:( 
                time.sleep(.1)
        # report list of A errors if there is one (there always is)
        if len(aberrantWs) > 0: 
            print(f"Aberrant work pages are #s {aberrantWs}")
        else: 
            print(f"There were no aberrant work pages!")

    def parseWs(self, html): 
        """This function retrieves Work info from the HTML provided"""
        
        # extract things we will need
        content = html.find('div', {'class':'col-md-9 fixed-height'})
        # try to locate author text of Work
        try: 
            author = content.div.h3.text
        except: 
            author = "unknown"
        # try to locate title text of Work
        try: 
            title = content.div.h4.text
        except: 
            title = "untitled"
        # try for both stanza and para text because they return [] 
        stanza_text = content.find_all('p',{'class':'stanza'})
        prose_text = content.find_all('p',{'class':'text'})
        # decide which text to use based on length 
        if len(prose_text) > len(stanza_text): 
            text = prose_text
            genre = 'prose'
        else: 
            text = stanza_text
            genre = 'poetry'
        # get actual text 
        Wtext = [x.text.replace('\n','').strip() for x in text]
        # make list of keys of Work types
        typeKeys = [x.text[:-1] for x in html.find('div', {'class':'card-body'}).find_all('h4')]
        # make list of values of Work types 
        typeVals = [x.text for x in html.find('div', {'class':'card-body'}).find_all('p')]
        # make dictionary of keys:values from above
        typeDict = dict(zip(typeKeys, typeVals))
        # initialize sub dictionary of Work
        Wdict = {'title': title, 
                   'genre': genre,
                   'text': Wtext} 
        # add type keys and values to complete sub dictionary of Work
        Wdict.update(typeDict) 
        # return Work dict to be made into a DataFrame row
        return Wdict

    def parseAs(self, html): 
        """This function retrieves Agent info from the HTML provided"""
        
        # get Agent's name
        name = html.find('div', {'class': 'card scrollable'}).h2.text 
        # get Agent's birth- and deathdates 
        bdate, ddate = html.find('div', {'class': 'card scrollable'}).span.text.split(' - ') 
        # initialize dictionary of Agent
        Adict = {'name': name, 'birth': bdate, 'death': ddate}
        # make list of type keys
        typeKeys = [x.h4.text.lower().replace(' ','_') for x in html.find_all('div', {'class': 'col-md-4'})]
        # initialize list of type values
        typeVals = []
        # add type values to list if found or default to 'unknown' (though some real values are also 'unknown')
        for typ in html.find_all('div', {'class': 'col-md-4'}): 
            try: 
                typeVals.append(typ.p or typ.div.span.text)
            except: 
                typeVals.append("unknown")
        # for some reason keys are at different levels and require '.text.' attribute but of course some don't
        typeVals = [x.text if not isinstance(x, str) else x for x in typeVals]
        # make dictionary of keys:values from above
        typeDict = dict(zip(typeKeys, typeVals))
        # add type keys and values to complete sub dictionary of Work
        Adict.update(typeDict) 
        # return Agent dict to be made into a DataFrame row 
        return Adict
    
    def get_single(self, cat, id_num): 
        """This function combines the functions above and returns the DataFrame row""" 
        
        # make url from parameters and initialize request 
        with requests.get(domain+cat+'s/'+str(id_num), verify=False) as r: 
            # double check the URL is correct
            print(r.url)
            # check status code
            print(f"{cat+str(id_num)} status code: {r.status_code}")
            # if connection is successful
            if r.status_code == 200: 
                # make soup from html content 
                s = BeautifulSoup(r.content, 'html.parser')
                # sort by FOO type 
                if cat.lower() == 'work': 
                    # make dictionary if Work with parseWs function
                    SubDict = {id_num:self.parseWs(s)}
                elif cat.lower() == 'agent': 
                    # make dictionary if Agent with parseAs function
                    SubDict = {id_num:self.parseAs(s)} 
                else: 
                    print("You need a category: 'work' or 'agent'...") 
                # make DataFrame row from dictionary
                singleDf = pd.DataFrame.from_dict(newSubDict, orient='index')
                # return DataFrame row
                return singleDf
            # return status code if connection is unsuccessful
            else: 
                print(f"Error: {r.status_code}") 
                
        def save_obj(obj, name):
            with open(name + '.pkl', 'wb+') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        def load_obj(name):
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)


    def run(self): 
        """This function runs retrieval and parsing using the functions above, creates DataFrames, and persists them (JSON)"""
        
        logging.info(f"Getting As and Ws")
        print(f"Getting As")
        # get starting time for A retrieval
        at1 = time.time()
        # retrieve As
        self.get_htmlA() 
        # get finishing time for A retrieval
        at2 = time.time()
        # display run times for A retrieval 
        logging.info(f"Got As in {round((at2-at1), 3)} sec ({round(((at2-at1)/list(tables.values())[0]), 5)} sec/ea.)")
        
        print(f"Getting Ws")
        # get starting time for W retrieval
        wt1 = time.time()
        # retrieve As
        self.get_htmlW() 
        # get finishing time for W retrieval
        wt2 = time.time()
        # display run times for W retrieval 
        print(f"Got Ws in {round((wt2-wt1), 3)} sec ({round(((wt2-wt1)/list(tables.values())[1]), 5)} sec/ea.)")
        logging.info(f"Got Ws in {round((wt2-wt1), 3)} sec ({round(((wt2-wt1)/list(tables.values())[1]), 5)} sec/ea.)")
        
        logging.info(f"Done getting As and Ws")
        
        self.save_obj(self.Asoup, 'Asoup')
        self.save_obj(self.Wsoup, 'Wsoup')
        
        logging.info(f"Parsing As and Ws")
        
        logging.info(f"Parsing As")
        print(f"Parsing As")
        # get starting time for A parsing
        pa1 = time.time()
        # parse Agent HTML instances
        #self.As = {k: self.parseAs(v) for k, v in self.Asoup.items()}
        # get finishing time for A parsing
        pa2 = time.time()
        # display run times for A parsing 
        logging.info(f"Parsed As in {round((pa2-pa1), 3)} sec ({round((pa2-pa1)/len(list(self.Asoup.items())), 5)} sec/ea.)")
        
        print(f"Parsing Ws")
        # get starting time for W parsing
        pw1 = time.time()
        # parse Work HTML instances 
        #self.Ws = {k: self.parseWs(v) for k, v in self.Wsoup.items()}
        # get finishing time for W parsing
        pw2 = time.time() 
        # display run times for W parsing 
        print(f"Parsed Ws in {round((pw2-pw1), 3)} sec ({round((pw2-pw1)/len(list(self.Wsoup.items())), 5)} sec/ea.)")
        logging.info(f"Parsed Ws in {round((pw2-pw1), 3)} sec ({round((pw2-pw1)/len(list(self.Wsoup.items())), 5)} sec/ea.)")
        logging.info(f"Done parsing As and Ws")
        
        logging.info(f"Making dataframes")
        print(f"Making AsDf")
        # create DataFrame of Agent dictionaries
        #AsDf = pd.DataFrame.from_dict(self.As, orient='index')
        print(f"Making WsDf")
        # create DataFrame of Work dictionaries
        #WsDf = pd.DataFrame.from_dict(self.Ws, orient='index')  
        logging.info(f"Done making dataframes")
        
        logging.info(f"Writing to json")
        # write Work DataFrame to JSON
        #WsDf.to_json('WsDf.json')
        # write Agent DataFrame to JSON
        #AsDf.to_json('AsDf.json')
        logging.info(f"Done writing to json")

def check_status(urls): 
    aberrantURLs = []
    logging.info(f"Checking status of URLs")
    for url in urls: 
        logging.info(f"Trying {url}")
        with requests.get(url, verify=False) as r: 
            if r.status_code == 200: 
                #logging.info(f"{url} successful")
                pass
            else: 
                logging.info(f"{url}: {r.status_code}")
                aberrantURLs.append({url: r.status_code})

class MAPRR: 
    
    def __init__(self): 
        self.urls_to_visit = []
        self.aberrantAs = []
        self.aberrantWs = []
        self.soups = {}
        self.Wsoup = {} 
        self.Asoup = {}
        self.Ws = {}
        self.As = {}
    
    def get_html(self, url): 
        url_format = 'https://mpgrr.herokuapp.com/(\w+)/(\d{1,3})'
        url_match = re.match(url_format, url)
        fco_type = url_match.group(1)
        id_num = url_match.group(2)
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    }
        with requests.get(url, headers=headers) as r: 
            logging.info(f"{fco_type}/{id_num} status code: {r.status_code}")
            if r.status_code == 200: 
                s = BeautifulSoup(r.content, 'html.parser')
                #self.soups.update({id_num:s})
                if fco_type == 'agents': 
                    self.Asoup.update({id_num:s})
                elif fco_type == 'works': 
                    self.Wsoup.update({id_num:s})
            else: 
                if fco_type == 'agents': 
                    self.aberrantAs.append({'A'+str(id_num): r.status_code})
                elif fco_type == 'works': 
                    self.aberrantWs.append({'W'+str(id_num): r.status_code})
                pass
        time.sleep(.5)
        
    def parse_html(self, html): 
        if 'works' in list(html.body.attrs.values())[0]: 
            content = html.find('div', {'class':'col-md-9 fixed-height'})
            try: 
                author = content.div.h3.text
            except: 
                author = ""
            try: 
                title = content.div.h4.text
            except: 
                title = ""
            stanza_text = content.find_all('p',{'class':'stanza'})
            prose_text = content.find_all('p',{'class':'text'})
            if len(stanza_text) > len(prose_text): 
                text = stanza_text
            elif len(stanza_text) < len(prose_text): 
                text = prose_text
            Wtext = [x.text.replace('\n','').strip() for x in text]
            metaKeys = [x.text[:-1] for x in html.find('div', {'class':'card-body'}).find_all('h4')]
            metaVals = [x.text for x in html.find('div', {'class':'card-body'}).find_all('p')]
            metaDict = dict(zip(metaKeys, metaVals))
            subDict = {'title': title, 
                       'text': Wtext}

            #self.Ws.update(subdict)
            return subDict

        elif 'agents' in list(html.body.attrs.values())[0]: 
            name = html.find('div', {'class': 'card scrollable'}).h2.text
            bdate, ddate = html.find('div', {'class': 'card scrollable'}).span.text.split(' - ')
            subDict = {'name': name, 'birth': bdate, 'death': ddate}
            try: 
                typeKeys = [x.h4.text for x in html.find_all('div', {'class': 'col-md-4'})]
                typeVals = [x.p or x.div.span.text for x in html.find_all('div', {'class': 'col-md-4'})]
                typeVals[:2] = [x.text for x in typeVals[:2]]
                typeDict = dict(zip(typeKeys, typeVals))
                subDict.update(typeDict)
            except: 
                pass

            #self.As.update(subdict)
            return subDict

    def run(self): 
        for t, i in list(tables.items())[:2]: 
            for j in range(0,5): 
                self.urls_to_visit.append(domain+t+str(j))
        
        logging.info(f"Getting As and Ws")
        for url in self.urls_to_visit: 
            self.get_html(url) 
        logging.info(f"Done getting As and Ws")
        
        logging.info(f"Parsing As and Ws")
        print(f"Parsing As")
        self.As = {k: self.parse_html(v) for k, v in self.Asoup.items()}
        print(f"Parsing Ws")
        self.Ws = {k: self.parse_html(v) for k, v in self.Wsoup.items()}        
        logging.info(f"Done parsing As and Ws")
        
        logging.info(f"Making dataframes")
        print(f"Making AsDf")
        AsDf = pd.DataFrame.from_dict(self.As, orient='index')
        print(f"Making WsDf")
        WsDf = pd.DataFrame.from_dict(self.Ws, orient='index')        
        logging.info(f"Done making dataframes")
        
        logging.info(f"Writing to json")
        WsDf.to_json('WsDf.json')
        AsDf.to_json('AsDf.json')
        logging.info(f"Done writing to json")
        
class ParallelMAPRR: 
    
    global domain, tables, max_threads
    
    def __init__(self): 
        self.urls_to_visit = []
        self.aberrantAs = []
        self.aberrantWs = []
        self.soups = {}
        self.Wsoup = {} 
        self.Asoup = {}
        self.Ws = {}
        self.As = {}
    
    def get_html(self, url): 
        url_format = 'https://mpgrr.herokuapp.com/(\w+)/(\d{1,3})'
        url_match = re.match(url_format, url)
        fco_type = url_match.group(1)
        id_num = url_match.group(2)
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    }
        with requests.get(url, headers=headers) as r: 
            logging.info(f"{fco_type}/{id_num} status code: {r.status_code}")
            if r.status_code == 200: 
                s = BeautifulSoup(r.content, 'html.parser')
                self.soups.update({id_num:s})
                #if fco_type == 'agents': 
                #    self.Asoup.update({id_num:s})
                #elif fco_type == 'works': 
                #    self.Wsoup.update({id_num:s})
            else: 
                if fco_type == 'agents': 
                    self.aberrantAs.append({'A'+str(id_num): r.status_code})
                elif fco_type == 'works': 
                    self.aberrantWs.append({'W'+str(id_num): r.status_code})
                pass
        time.sleep(.5)
            
    def downloadHTML(self): 
        threads = min(max_threads, len(self.urls_to_visit)) 
        print(f"Downloading with {threads} threads")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor: 
            executor.map(self.get_html, self.urls_to_visit)

    def parse_html(self, html): 
        if 'works' in list(html.body.attrs.values())[0]: 
            content = html.find('div', {'class':'col-md-9 fixed-height'})
            try: 
                author = content.div.h3.text
            except: 
                author = ""
            try: 
                title = content.div.h4.text
            except: 
                title = ""
            stanza_text = content.find_all('p',{'class':'stanza'})
            prose_text = content.find_all('p',{'class':'text'})
            if len(stanza_text) > len(prose_text): 
                text = stanza_text
            elif len(stanza_text) < len(prose_text): 
                text = prose_text
            Wtext = [x.text.replace('\n','').strip() for x in text]
            metaKeys = [x.text[:-1] for x in html.find('div', {'class':'card-body'}).find_all('h4')]
            metaVals = [x.text for x in html.find('div', {'class':'card-body'}).find_all('p')]
            metaDict = dict(zip(metaKeys, metaVals))
            subDict = {'title': title, 
                       'text': Wtext}

            self.Ws.update(subdict)
            #return subDict

        elif 'agents' in list(html.body.attrs.values())[0]: 
            name = html.find('div', {'class': 'card scrollable'}).h2.text
            bdate, ddate = html.find('div', {'class': 'card scrollable'}).span.text.split(' - ')
            subDict = {'name': name, 'birth': bdate, 'death': ddate}
            try: 
                typeKeys = [x.h4.text for x in html.find_all('div', {'class': 'col-md-4'})]
                typeVals = [x.p or x.div.span.text for x in html.find_all('div', {'class': 'col-md-4'})]
                typeVals[:2] = [x.text for x in typeVals[:2]]
                typeDict = dict(zip(typeKeys, typeVals))
                subDict.update(typeDict)
            except: 
                pass

            self.As.update(subdict)
            #return subDict
        
        else: 
            logging.info(f"Something went wrong while parsing")
            pass
    
    def parseHTML(self): 
        threads = min(max_threads, len(self.urls_to_visit)) 
        print(f"Parsing with {threads} threads")
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor: 
            executor.map(self.parse_html, self.soups)
    
    def run(self): 
        for t, i in list(tables.items())[:2]: 
            for j in range(0,5): 
                self.urls_to_visit.append(domain+t+str(j))
        
        logging.info(f"Getting As and Ws")
        self.downloadHTML()
        logging.info(f"Done getting As and Ws")
        
        logging.info(f"Parsing As and Ws")
        self.parseHTML()
        logging.info(f"Done parsing As and Ws")
        
        logging.info(f"Making dataframes")
        print(f"Making AsDf")
        AsDf = pd.DataFrame.from_dict(self.As)
        print(f"Making WsDf")
        WsDf = pd.DataFrame.from_dict(self.Ws)        
        logging.info(f"Done making dataframes")
        
        logging.info(f"Writing to json")
        WsDf.to_json('WsDf.json')
        AsDf.to_json('AsDf.json')
        logging.info(f"Done writing to json")