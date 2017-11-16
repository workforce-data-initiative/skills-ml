from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator

import json
from collections import Counter, OrderedDict, defaultdict

import csv

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import nltk
import re



try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

f = open('VA_caac4ea4-809b-4938-b339-8c959f5c3c81','r')
skills_filename = "output/skills_master_table.tsv"

#out = open("NLX_2011Q1_description.txt", "wb")

def reg_ex(s):
    s = s.replace(".","\.")
    s = s.replace("^","\^")
    s = s.replace("$","\$")
    s = s.replace("*","\*")
    s = s.replace("+","\+")
    s = s.replace("?","\?")
    return s

def skills_lookup(skills_filename):
    with open(skills_filename) as infile:
        reader = csv.reader(infile, delimiter='\t')
        header = next(reader)
        index=3
        generator = (reg_ex(row[index]) for row in reader)
        
        return set(generator)

def ie_preprocess(document):
    """This function takes raw text and chops and then connects the process to break     
       it down into sentences"""
    
    # Pre-processing
    # e.g.","exempli gratia"
    document=document.replace("e.g.","exempli gratia")
    
    # Sentence tokenizer out of nltk.sent_tokenize
    split = re.split('\n|\*', document)
    
    # Sentence tokenizer
    sentences=[]
    for sent in split:
        sents = nltk.sent_tokenize(sent)
        length = len(sents)
        if length == 0:
            next
        elif length == 1:
            sentences.append(sents[0])
        else:
            for i in range(length):
                sentences.append(sents[i])
    return sentences


def generate_candidates(job_postings):
    available_skills = skills_lookup(
    corpus_creator = SimpleCorpusCreator()
    corpus_creator.document_schema_fields = ['description']
    for line in job_postings:
        data = json.loads(line, object_pairs_hook=OrderedDict)
        document = corpus_creator._join(data)
        sentences = ie_preprocess(document)
        
        skills = defaultdict(list)
        num=0
        for skill in available_skills:
            len_skill = len(skill.split())
            for sent in sentences:
                sent = sent.encode('utf-8')
                
                #Exact matching
                if len_skill ==1:
                    sent = sent.decode('utf-8')
                    if re.search(r'\b'+skill +r'\b', sent, re.IGNORECASE):
                        skills[skill] = [100]
                        skills[skill].append(sent)
                #Fuzzy matching
                else:        
                    ratio = fuzz.partial_ratio(skill,sent)
                    # You can adjust the partial of matching here: 100 => exact matching 0 => no matching
                    if ratio > 88:
                        skills[skill] = [ratio]
                        skills[skill].append(sent)
            
            num +=1
        for key, values in skills.items():
            for v in values:
                try:
                    print (key," : ", v)
                except:
                    print (key," : ", v.encode('utf-8'))

