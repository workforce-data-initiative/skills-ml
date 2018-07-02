"""Normalize a job title through Explicit Semantic Analysis

Originally written by Kwame Porter Robinson
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csr import csr_matrix
from nltk.corpus import wordnet as wn
import itertools
import nltk
import re
import numpy as np
import logging
from skills_ml.datasets.onet_source import OnetToDiskDownloader

ONET_VERSIONS = [
    'db_21_0_text',
    'db_20_3_text',
    'db_20_2_text',
    'db_20_1_text',
    'db_20_0',
    'db_19_0',
    'db_18_1',
    'db_18_0',
    'db_17_0',
    'db_16_0',
    'db_15_1',
    'db_15_0',
    'db_14_0',
    'db_13_0',
    'db_12_0',
    'db_11_0',
    'db_10_0',
]



class ESANormalizer(object):
    """Normalize a job title to ONET occupation titles using explicit semantic analysis.

    Uses ONET occupation titles and descriptions.
    """
    def __init__(self, onet_source=OnetToDiskDownloader):
        self.onet_downloader = onet_source()
        self.onet_titles = self.retrieve_onet_titles()
        logging.info('Retrieved onet titles')
        # ... Following the ESA description:
        # https://en.wikipedia.org/wiki/Explicit_semantic_analysis
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        # optimization note: convert from CSR to CSC
        self.tf = self.tfidf_vectorizer.fit_transform(self.onet_titles['Description'].values)
        self.concept_row = self.onet_titles.index.values
        try:
            wn.synset
        except LookupError:
            nltk.download('wordnet')

    def retrieve_onet_titles(self):
        onet_titles = pd.concat(
            (pd.read_csv(self.onet_downloader.download(
                version,
                'Occupation Data.txt',
                'occupation_data.txt'
            ), sep='\t') for version in ONET_VERSIONS),
            ignore_index=True
        )
        # Assumes pandas 0.19, keeps newest duplicate Title
        onet_titles.drop_duplicates('Title', inplace=True, keep='last')
        onet_titles['Major'] = onet_titles.iloc[:, 0].apply(lambda x: x[:2])

        LOWER = True
        if LOWER:
            # all RDD strings are unicode
            onet_titles['Title'] = onet_titles['Title'].str.lower()
            onet_titles['Description'] = onet_titles['Description'].str.lower()

        # now we can do a title -> Major, Minor lookup
        onet_titles.set_index('Title', inplace=True)
        # access with onet_titles.loc[u'Sales Agents, Financial Services']
        return onet_titles

    def hypernym_product(self, job_title):
        word_hypernyms = []
        for word in job_title.split():
            synset = wn.synsets(word)
            if synset:
                hypernyms = synset[0].hypernym_paths()[0][-3:]
                word_hypernyms.append(
                    [re.split('_|\.', hypernym.name())[0] for hypernym in hypernyms]
                )

        for hypernym_title in itertools.product(*word_hypernyms):
            yield hypernym_title

    def concept_rank(self, words):
        """
        Attempt to normalize this title to ONET Occupation treated as 'concepts'
        """
        concepts = csr_matrix((len(self.concept_row), 1))
        for word in words:
            if word in self.tfidf_vectorizer.vocabulary_:
                concept_vector_idx = self.tfidf_vectorizer.vocabulary_[word]
                concept_vector = self.tf[:, concept_vector_idx]

                concepts += concept_vector

        # Output ranked list of concepts, ONET Occupation, truncated to 3 for now
        ranked_concepts = concepts.nonzero()[0]
        for idx in np.argsort(ranked_concepts)[:3]:
            concept_idx = ranked_concepts[idx]
            concept = self.concept_row[concept_idx]
            concept_tfidf = concepts[concept_idx, 0]
            # print("Concept/Occupation: {}, tf/idf: {}, index: {}".format(concept,
            #                                                             concept_tfidf,
            #                                                             concept_idx))
            yield {'title': concept, 'relevance_score': concept_tfidf}


    def normalize_job_title(self, job_title):
        normalized_titles = []
        seen = set()  # doubles memory, may want to hash
        seen_add = seen.add

        ret = []
        for title in self.hypernym_product(job_title):
            for concept in self.concept_rank(title):
                #logging.warning(concept)
                normalized_titles.append(concept)
        for item in sorted(normalized_titles,
                           key=lambda x: x['relevance_score'],
                           reverse=True):
            if hash(item['title']) not in seen:  # to conserve seen set memory
                seen_add(hash(item['title']))
                ret.append(item)
        return ret
