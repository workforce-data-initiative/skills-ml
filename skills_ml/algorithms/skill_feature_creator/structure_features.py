from nltk.tokenize import sent_tokenize 

import json
import csv
from collections import OrderedDict

import re
    
def structFeatures(sent, i, desc_length):
    
    structFeat =[]
    # structFeat 1. Position
    # 
    # 1:begining, 
    # 2:ending, 
    # 3:0-20%(except for beginning),
    # 4:21-40%
    # 5:41-60%
    # 6: 61-80%
    # 7: 81-100% (except for ending)
    
    position = i*100/desc_length
    
    if i==0:
       structFeat.append(1) 
    elif i==desc_length-1:
       structFeat.append(2) 
    elif 0 < position <= 20:
       structFeat.append(3)
    elif 20 < position <= 40:
       structFeat.append(4)
    elif 40 < position <= 60:
       structFeat.append(5)
    elif 60 < position <= 80:
       structFeat.append(6)
    else:
       structFeat.append(7)
    
    # structFeat 2. start_with_a_symbol + * -
    # 
    # 1:Yes, 
    # 0:No
    
    symbols = ['+','-','*']
    start = sent.split(' ',1)[0]
    start = start.replace('\n','')
    
    if start in symbols:
       structFeat.append(1)
    else:
       structFeat.append(0)
    
    return structFeat

# Sentence Tokenization
def sentTokenize(jobDescription):
    if '\n' in description:
        sentences = re.split('\n', description)
    else:
        try:
            sentences = nltk.sent_tokenizes(description.encode('utf-8'))
        except:
            sentences = nltk.sent_tokenize(description)
    return sentences

def structFeatGeneration(jobPosting):
    
    data = json.loads(jobPosting, object_pairs_hook=OrderedDict)
    try:
        description = data["description"]
    except:
        description = data[""]
    
    sentences = sentTokenize(description)
    desc_length = len(sentences)
    features = [structFeatures(sentences[i], i, desc_length) for i in range(desc_length)]
    
    return features