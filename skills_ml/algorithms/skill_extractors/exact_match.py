"""Use exact matching with a source list to find skills"""

import unicodecsv as csv
import logging
from collections import Counter
from smart_open import smart_open
import re

import nltk
try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

from .base import CandidateSkill, ListBasedSkillExtractor


class ExactMatchSkillExtractor(ListBasedSkillExtractor):
    """Extract skills from unstructured text

    Originally written by Kwame Porter Robinson
    """
    name = 'exact'

    def _skills_lookup(self):
        """Create skills lookup

        Reads the object's filename containing skills into a lookup

        Returns: (set) skill names
        """
        logging.info('Creating skills lookup from %s', self.skill_lookup_path)
        with smart_open(self.skill_lookup_path) as infile:
            reader = csv.reader(infile, delimiter='\t')
            header = next(reader)
            index = header.index(self.nlp.transforms[0])
            generator = (row[index] for row in reader)
            return set(generator)

    def _document_skills_in_lookup(self, document, lookup):
        """Count skills in the document

        Args:
            lookup (object) A collection that can be queried for an individual skill,
                implementing 'in' (i.e. 'skill in lookup')
            document (string) A document for searching, such as a job posting

        Returns: (collections.Counter) skills present in the lookup found in the document
            All values set to 1 (multiple occurrences of a skill do not count)
        """
        join_spaces = " ".join  # for runtime efficiency
        N = 5
        doc = document.split()
        doc_len = len(doc)
        skills = Counter()

        start_idx = 0

        while start_idx < doc_len:
            offset = 1

            lookahead = min(N, doc_len - start_idx)
            for idx in range(lookahead, 0, -1):
                ngram = join_spaces(doc[start_idx:start_idx+idx])
                if ngram in lookup:
                    skills[ngram] = 1
                    offset = idx
                    break

            start_idx += offset
        return skills

    def candidate_skills(self, job_posting):
        document = job_posting.text
        sentences = self.ie_preprocess(document)

        for skill in self.lookup:
            len_skill = len(skill.split())
            for sent in sentences:
                sent = sent.encode('utf-8')

                # Exact matching for len(skill) == 1
                if len_skill == 1:
                    sent = sent.decode('utf-8')
                    if re.search(r'\b' + skill + r'\b', sent, re.IGNORECASE):
                        yield CandidateSkill(
                            skill_name=skill,
                            matched_skill=skill,
                            confidence=100,
                            context=sent
                        )
