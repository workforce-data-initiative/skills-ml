from string_cleaners import NLPTransforms
from corpus_creators.basic import SimpleCorpusCreator

import json
from collections import Counter, OrderedDict, defaultdict

import csv

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import nltk
import re



f = open('random200_NLX_b8384025-fa09-417d-ae10-96880fac86be','r')
skills_filename = "skills_master_table.tsv"

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
    
    #QUESTION
    #index = header.index(self.nlp.transforms[0])
    
    #find longest list
    #long_l = [row[index] for row in reader]
    #print (max(long_l, key=len))
    #digital imaging communications in medicine dicom-compatible image acquisition and integration software products
    #for row in reader:
     #   skill = row[3]
      #  words = skill.split()
       # result = skill+"\t"+str(len(words))
        #print(result)   

    #generator = (reg_ex(row[index]) for row in reader)

    #lookup = set[generator]


def ie_preprocess(document):
    """This function takes raw text and chops and then connects the process to break     
       it down into sentences"""
    
    # Pre-processing
    # e.g.","exempli gratia"
    document=document.replace("e.g.","exempli gratia")
    
    # Sentence tokenizer out of nltk.sent_tokenize
    split = re.split('\n|\*', document)
    
    # Sentence tokenizer
    #sentences = [nltk.sent_tokenize(sent) for sent in split]
    sentences=[]
    for sent in split:
        sents = nltk.sent_tokenize(sent)
        length = len(sents)
        if length ==0:
            next
        elif length == 1:
            sentences.append(sents[0])
        else:
            for i in range(length):
                sentences.append(sents[i])
                
            
            
        
    
    #try:
     #   sentences = nltk.sent_tokenize(document.encode('utf-8'))
    #except:
     #   sentences = nltk.sent_tokenize(document)
    
    #sentences = [nltk.word_tokenize(sent) for sent in sentences]
    #sentences = [nltk.wordpunct_tokenize(sent) for sent in sentences]
    
    #sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

for line in f:

    data = json.loads(line, object_pairs_hook=OrderedDict)
        
    #SimpleCorpusCreator
    #document = SimpleCorpusCreator()._transform(data)
    document = SimpleCorpusCreator()._join(data)
    
    
    #document = document.replace("\n", ". ")
    #document = document.replace("*", ". *")
    sentences = ie_preprocess(document)
    
    #skills = Counter()
    skills = defaultdict(list)
    num=0
    for skill in skills_lookup(skills_filename):
    #for skill in lookup:
    
        
        #print(str(num))
        #print(skill)
        len_skill = len(skill.split())
        for sent in sentences:
            sent = sent.encode('utf-8')
            #print(skill+"\t"+sent)
            #sent = sent.encode('utf-8')
            
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
    #print ("DONE!")
    #print(skills)
    for key, values in skills.items():
        for v in values:
            try:
                print (key," : ", v)
            except:
                print (key," : ", v.encode('utf-8'))

    
    

    
    #out.write(id_as_bytes)    
    #out.write(description)
    #print (description)
#out.close()



    

