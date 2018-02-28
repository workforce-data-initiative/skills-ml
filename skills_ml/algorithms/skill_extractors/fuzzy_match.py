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

from .base import CandidateSkill, ListBasedSkillExtractor


class FuzzyMatchSkillExtractor(ListBasedSkillExtractor):
    """Extract skills from unstructured text using fuzzy matching"""
    name = 'fuzzy'
    match_threshold = 88

    def reg_ex(self, s):
        s = s.replace(".", "\.")
        s = s.replace("^", "\^")
        s = s.replace("$", "\$")
        s = s.replace("*", "\*")
        s = s.replace("+", "\+")
        s = s.replace("?", "\?")
        return s

    def _skills_lookup(self):
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

    def candidate_skills(self, job_posting):
        document = job_posting.text
        sentences = self.ie_preprocess(document)

        for skill in self.lookup:
            len_skill = len(skill.split())
            for sent in sentences:
                sent = sent.encode('utf-8')

                # Exact matching
                if len_skill == 1:
                    sent = sent.decode('utf-8')
                    if re.search(r'\b' + skill + r'\b', sent, re.IGNORECASE):
                        logging.info('Returning exact match %s in sent %s', skill, sent)
                        yield CandidateSkill(
                            skill_name=skill,
                            matched_skill=skill,
                            confidence=100,
                            context=sent
                        )
                # Fuzzy matching
                else:
                    ratio = fuzz.partial_ratio(skill, sent)
                    # You can adjust the partial of matching here:
                    # 100 => exact matching 0 => no matching
                    if ratio > self.match_threshold:
                        logging.info('Found fuzzy matches passing threshold in %s', sent)
                        for match in self.fuzzy_matches_in_sentence(skill, sent):
                            logging.info('Returning fuzzy match %s in sent: %s', match, sent)
                            yield match
