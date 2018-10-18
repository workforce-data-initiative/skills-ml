# coding: utf-8
from skills_ml.algorithms.skill_feature_creator.posTags import tagMatching
from skills_ml.algorithms.nlp import sentence_tokenize
from functools import reduce
from operator import add
import nltk
from nltk import ngrams

import json
import csv
from collections import OrderedDict

import re

def is_upper(word):
    if word.isupper() == True:
        return 1
    else:
        return 0

def is_title(word):
    if word.istitle() == True:
        return 1
    else:
        return 0

def is_digit(word):
    if word.isdigit() == True:
        return 1
    else:
        return 0

def wordPos(i,length):
    if i==0:
        return 0
    elif i == length-1:
        return 1
    else:
        return 2


def word2features(sent, i):
    localContFeat =[]

    word = sent[i][0]
    postag = sent[i][1]
    length = len(sent)

    # localContFeat 00. is_Skill or is_Experience?
    # Skill in 3 grams: 1
    # Experience in 3 grams: 2
    # Ability in 3 grams: 3
    # Else: 0
    if i < length-2:
        threegrams = sent[i][0] + " " + sent[i+1][0] + " "+ sent[i+2][0]
        if "skill" in threegrams:
            localContFeat.append(1)
        elif "experience" in threegrams:
            localContFeat.append(1)
        elif "ability" in threegrams:
            localContFeat.append(1)
        else:
            localContFeat.append(0)
    elif i >= length-2:
        localContFeat.append(0)

    # localContFeat 0. Unigram
    # Uncomment this laster if needed
    # localContFeat.append(word)

    # localContFeat 1. Unigram.lower()
    # Uncomment this laster if needed
    # localContFeat.append(word.encode('utf-8').lower())

    # localContFeat 2. Case
    # All capital: 1
    # Else: 0
    localContFeat.append(is_upper(word))

    # localContFeat 3. First character case
    # Capital: 1
    # Else: 0
    localContFeat.append(is_title(word))

    # localContFeat 4. Digit
    # Digit: 1
    # Else: 0
    localContFeat.append(is_digit(word))

    # localContFeat 5. POS
    localContFeat.append(tagMatching(postag))

    # localContFeat 6. POS - big category
    localContFeat.append(tagMatching(postag[:2]))

    # localContFeat 7. Word Position
    # Beginning: 0
    # Ending: 1
    # Else: 2
    localContFeat.append(wordPos(i,length))

    # localContFeat 8 - 12. Previous word, First character case, All upper case, POS, POS - big category
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]

        # Previous word: Uncomment this if needed
        #localContFeat.append(word1.encode('utf-8').lower())
        localContFeat.append(is_upper(word1))
        localContFeat.append(is_title(word1))
        localContFeat.append(is_digit(word1))
        localContFeat.append(tagMatching(postag1))
        localContFeat.append(tagMatching(postag1[:2]))
    # frist word
    else:
        localContFeat.extend([-1]*5)

    # localContFeat 13 - 17. Next word, First character case, All upper case, POS, POS - big category
    if i < length-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]

        # Next word: Uncomment this if needed
        #localContFeat.append(word1.encode('utf-8').lower())
        localContFeat.append(is_upper(word1))
        localContFeat.append(is_title(word1))
        localContFeat.append(is_digit(word1))
        localContFeat.append(tagMatching(postag1))
        localContFeat.append(tagMatching(postag1[:2]))
    # last word
    else:
        localContFeat.extend([-1]*5)

    return localContFeat

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def pre_process(sentences, word_tokenizer):
    # """This function takes raw text and chops and then connects the process to break
    #    it down into sentences, then words and then complete part-of-speech tagging"""

    # Break sentences into words with punctuations
    sentences = [word_tokenizer(sent) for sent in sentences]
    # Break sentences into words without punctuations
    #sentences = [nltk.wordpunct_tokenize(sent) for sent in sentences]

    # Tag each word with its POS (Part-Of-Speech) label
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences

def local_contextual_features(job_posting):
    sentences = sentence_tokenize(job_posting)
    sentences = pre_process(sentences)
    features = [sent2features(s) for s in sentences]
    return features

