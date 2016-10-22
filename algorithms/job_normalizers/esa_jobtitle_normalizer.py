import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix
from nltk.corpus import wordnet as wn
import itertools
import json
import nltk
import re
import numpy as np
import glob
import os

# Download content for query expansion
try:
    wn.synset
except LookupError:
    nltk.download('wordnet')

# Simple attempt to use Explicit Semantic Analysis
# with ONET database occupation descriptions as a way of
# inferring job title similiarity

# Extract

# Read in Occupations, strip duplicates, create mapping tables
onet_occuptation_file = "Occupation Data.txt"
all_occupation_files = glob.glob('db_**_*/' + onet_occuptation_file)
onet_titles = pd.concat((pd.read_csv(f, sep='\t') for f in all_occupation_files),
                         ignore_index=True)

# Assumes pandas 0.19, keeps newest duplicate Title
onet_titles.drop_duplicates('Title', inplace=True, keep='last')
onet_titles['Major'] = onet_titles.iloc[:,0].apply(lambda x: x[:2])

LOWER = True
if LOWER:
      onet_titles['Title'] = onet_titles['Title'].str.lower() # all RDD strings are unicode
      onet_titles['Description'] = onet_titles['Description'].str.lower() 

onet_titles.set_index('Title', inplace=True) # now we can do a title -> Major, Minor lookup
# access with onet_titles.loc[u'Sales Agents, Financial Services']

# Transform
# ... Following the ESA description: https://en.wikipedia.org/wiki/Explicit_semantic_analysis
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# optimization note: convert from CSR to CSC
tf = tfidf_vectorizer.fit_transform(onet_titles['Description'].values)
inverse_vocabulary = {v: k for k, v in tfidf_vectorizer.vocabulary_.items()} # for inspection, not required
concept_row = onet_titles.index.values

## quick check, first row maps words correclty
#pub_rel_words = [inverse_vocabulary[x] for x in cols]

# Load
# .... Using the above ESA representation of ONET Occupations
# we use concept ranking and hypernyms to generate normalized job titles

def hypernym_product(job_title):
    word_hypernyms = []
    for word in job_title.split():
        synset = wn.synsets(word)
        if synset:
            hypernyms = synset[0].hypernym_paths()[0][-3:]
            word_hypernyms.append(
                    [re.split('_|\.', hypernym.name())[0]
                                for hypernym in hypernyms] )

    for hypernym_title in itertools.product( *word_hypernyms ):
        yield hypernym_title

def concept_rank(words):
    """
    Attempt to normalize this title to ONET Occupation treated as 'concepts'
    """
    concepts = csr_matrix((len(concept_row), 1))
    for word in words:
        if word in tfidf_vectorizer.vocabulary_:
            concept_vector_idx = tfidf_vectorizer.vocabulary_[word]
            concept_vector = tf[:, concept_vector_idx]

            concepts += concept_vector # sum concept vectors

    # Output ranked list of concepts, ONET Occupation, truncated to 3 for now
    ranked_concepts = concepts.nonzero()[0]
    for idx in np.argsort( ranked_concepts )[:3]:
        concept_idx = ranked_concepts[idx]
        concept = concept_row[concept_idx]
        concept_tfidf = concepts[concept_idx, 0]
        #print("Concept/Occupation: {}, tf/idf: {}, index: {}".format(concept,
        #                                                             concept_tfidf,
        #                                                             concept_idx))
        yield {'title': concept,
                'relevance_score': concept_tfidf}

def normalize_job_title(job_title):
    normalized_titles = []
    seen = set() # doubles memory, may want to hash
    seen_add = seen.add

    ret = []

    for title in hypernym_product(job_title):
        for concept in concept_rank(title):
            #yield concept, title
            normalized_titles.append( concept )
    for item in sorted(normalized_titles,
                       key=lambda x: x['relevance_score'],
                       reverse=True):
        if hash(item['title']) not in seen: # to conserve seen set memory
            seen_add( hash(item['title']) )
            ret.append(item)

    return ret
