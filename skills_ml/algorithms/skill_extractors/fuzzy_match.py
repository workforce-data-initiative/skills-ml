"""Use fuzzy matching with a source list to extract skills from a job posting"""
import unicodecsv as csv
import logging
from smart_open import smart_open
import re

import nltk
try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

from fuzzywuzzy import fuzz

from .base import CandidateSkill, ListBasedSkillExtractor, CandidateSkillYielder, trie_regex_from_words
from typing import Dict, Text


class FuzzyMatchSkillExtractor(ListBasedSkillExtractor):
    """Extract skills from unstructured text using fuzzy matching"""

    match_threshold = 88

    @property
    def method_name(self) -> Text:
        return f'fuzzy_{self.match_threshold}'

    @property
    def method_description(self) -> Text:
        return f'Fuzzy matching using ratio of most similar substring, with a minimum cutoff of {self.match_threshold} percent match'

    def reg_ex(self, s):
        s = s.replace(".", "\.")
        s = s.replace("^", "\^")
        s = s.replace("$", "\$")
        s = s.replace("*", "\*")
        s = s.replace("+", "\+")
        s = s.replace("?", "\?")
        return s

    def _skills_lookup(self) -> set:
        """Create skills lookup

        Reads the object's filename containing skills into a lookup

        Returns: (set) skill names
        """
        with smart_open(self.skill_lookup_path) as infile:
            reader = csv.reader(infile, delimiter='\t')
            next(reader)
            index = 3
            generator = (self.reg_ex(row[index]) for row in reader)

            return set(generator)

    def ngrams(self, sent, n):
        sent_input = sent.split()
        output = []
        for i in range(len(sent_input)-n+1):
            output.append(sent_input[i:i+n])
        return output

    def fuzzy_matches_in_sentence(self, skill, sentence):
        N = len(skill.split())
        doc = self.ngrams(sentence, N)
        doc_join = [b" ".join(d) for d in doc]

        for dj in doc_join:
            ratio = fuzz.partial_ratio(skill, dj)
            if ratio > self.match_threshold:
                yield CandidateSkill(
                    skill_name=skill,
                    matched_skill=dj,
                    confidence=ratio,
                    context=sentence.decode('utf-8')
                )

    def candidate_skills(self, source_object: Dict) -> CandidateSkillYielder:
        document = self.transform_func(source_object)
        sentences = self.nlp.sentence_tokenize(document)

        for sent in sentences:
            exact_matches = set()
            for cs in self.candidate_skills_in_context(sent): 
                yield cs
                if cs.skill_name not in exact_matches:
                    exact_matches.add(cs.skill_name)

            for skill in self.lookup:
                if skill in exact_matches:
                    continue
                ratio = fuzz.partial_ratio(skill, sent)
                # You can adjust the partial of matching here:
                # 100 => exact matching 0 => no matching
                if ratio > self.match_threshold:
                    logging.info('Found fuzzy matches passing threshold in %s', sent)
                    for match in self.fuzzy_matches_in_sentence(skill, sent.encode('utf-8')):
                        logging.info('Returning fuzzy match %s in sent: %s', match, sent)
                        yield match
