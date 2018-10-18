import json
import csv
from collections import OrderedDict
from skills_ml.algorithms.nlp import sentence_tokenize
import re

def struct_features(sent, i, desc_length, word_tokenizer):

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

    return [structFeat]*len(word_tokenizer(sent))


def structFeatGeneration(job_posting):
    sentences = sentence_tokenize(job_posting)
    desc_length = len(sentences)
    features = [struct_features(sentences[i], i, desc_length) for i in range(desc_length)]

    return features
