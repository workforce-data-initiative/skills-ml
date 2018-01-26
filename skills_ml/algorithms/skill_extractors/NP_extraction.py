# coding: utf-8
import json
from collections import OrderedDict
import io
import re
from collections import defaultdict
import operator

import nltk
import csv
from operator import itemgetter

import time

import collections

f = open('NLX_b8384025-fa09-417d-ae10-96880fac86be','r')
#f = open('NLX_100','r')

np_dic={}


def np_extraction(candidates):
    for c in candidates:
        c.strip()
        if c != "\n":
            #print(c.encode("utf-8"))
            output = ie_preprocess(c)
            
            grammar = r"""
                      NP: {<JJ.*>*<NN.*><NN.*>*}   # chunk adjectives and nouns
                       """
            cp = nltk.RegexpParser(grammar)
            for sent in output:
                #result = cp.parse(sent)
                #tree = cp.parse(result)
                if sent:
                    tree = cp.parse(sent)
                    for subtree in tree.subtrees():
                        if subtree.label() == 'NP': 
                            
                            np=""
        
                            for node in subtree:
                                try:
                                    np += node[0].encode('utf-8') + ' '
                                except:
                                    np += node[0] + ' '
                            np.strip()
                            #s = `id_cnt` +"\t" + np
                            #print np
                            
                            if not np in np_dic:
                                np_dic.update({np:1})
                            else:
                                np_dic[np] +=1
                                
def np_extraction2(c):

    c.strip()
    if c != "\n":
        #print(c.encode("utf-8"))
        output = ie_preprocess(c)
        
        grammar = r"""
                  NP: {<JJ.*>*<NN.*><NN.*>*}   # chunk adjectives and nouns
                   """
        cp = nltk.RegexpParser(grammar)
        for sent in output:
            #result = cp.parse(sent)
            #tree = cp.parse(result)
            if sent:
                tree = cp.parse(sent)
                for subtree in tree.subtrees():
                    if subtree.label() == 'NP': 
                        
                        np=""
    
                        for node in subtree:
                            try:
                                np += node[0].encode('utf-8') + ' '
                            except:
                                np += node[0] + ' '
                        np.strip()
                        #s = `id_cnt` +"\t" + np
                        #print np
                        
                        if not np in np_dic:
                            np_dic.update({np:1})
                        else:
                            np_dic[np] +=1


def ie_preprocess(document):
    """This function takes raw text and chops and then connects the process to break     
       it down into sentences, then words and then complete part-of-speech tagging"""
    try:
        sentences = nltk.sent_tokenize(document.encode('utf-8'))
    except:
        sentences = nltk.sent_tokenize(document)
    #sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.wordpunct_tokenize(sent) for sent in sentences]
    
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences




def first_word(desc_line):
    indicators={}
    sorted_indicators={}
    for c in desc_line:
        c.strip()
        if c != "\n":
            #indicators['first']=c[1]
            first=c.split(' ',1)[0]
            first = first.replace('\n','')
            if not first in indicators:
                indicators.update({first:1})
            else:
                indicators[first] +=1
    total = sum(indicators.values())
    sorted_indicators = sorted(indicators.items(), key = operator.itemgetter(1),reverse = True)
    try:
        most = sorted_indicators[0][0]
        most_value= sorted_indicators[0][1]
    except:
        most = 'Z'
    return most

total_dic={}
line_num=0

start = time.time()
print("Starting time: %.2f" %start)

c = collections.Counter()

for line in f:
    line_num +=1
    #print line_num
    
    data = json.loads(line, object_pairs_hook=OrderedDict)
    #keys=data.keys()
    job_id= data["id"]
    title = data["title"]
    soc = data["onet_soc_code"]
    soc_code = soc[:2]
    skills = data["skills"]
    #print(title)
    try:
        description = data["description"]
    except:
        description = data[""]
        #print(value)
        #print(description.encode('utf-8'))
    #description_list= description.split()
    

    #sentence tokenization
    desc_line = re.findall(r'\n.*', description)
    #desc_line = re.findall(r'\n', description)
    
    
    
    num=len(desc_line)
    #First Character Checking

    most_char=first_word(desc_line)
    bullet =['+', '*','-']
    for d in desc_line:
        d_first=d.split(' ',1)[0]
        d_first = d_first.replace('\n','')
        try:
            if d_first[0] in bullet:
                np_extraction2(d)
        except:
            next
        
    
    # NOUN PHRASE CHUNKING
    #np_extraction(desc_line)
end = time.time()
print ("Elasped time: %.2f" %(end-start))


with open('NLX_2011Q_all_out.csv', 'wb') as f:
#with open('NLX_100_out.csv', 'wb') as f:    
    w = csv.writer(f)
    for key, value in np_dic.items():
        w.writerow([key, value])

#for x in indicators:
#    print("%s : %d  " % (x.encode('utf-8'), indicators[x]))
#for x in np_dic:
#    try:
#        print("%s : %d  " % (x.encode('utf-8'), np_dic[x]))
#    except:
#        print("%s : %d  " % (x, np_dic[x]))