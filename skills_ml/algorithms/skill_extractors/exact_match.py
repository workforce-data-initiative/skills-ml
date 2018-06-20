"""Use exact matching with a source list to find skills"""

import unicodecsv as csv
import logging
from smart_open import smart_open
import re

import nltk
try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

from .base import CandidateSkill, ListBasedSkillExtractor, CandidateSkillYielder

from typing import Dict


class ExactMatchSkillExtractor(ListBasedSkillExtractor):
    """Extract skills from unstructured text

    Originally written by Kwame Porter Robinson
    """
    method_name = 'exact_match'
    method_description = 'Exact matching'

    def _skills_lookup(self) -> set:
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

    def candidate_skills(self, source_object:Dict) -> CandidateSkillYielder:
        """Yield objects which may represent skills/competencies from the given source object.

        Looks for exact matches between the reference skill lookup and the object's text.

        Args: source_object (dict) A structured document for searching, such as a job posting

        Yields: CandidateSkill objects
        """
        document = self.transform_func(source_object)
        sentences = self.nlp.sentence_tokenize(document)

        for skill in self.lookup:
            for sent in sentences:
                sent = sent.encode('utf-8')

                sent = sent.decode('utf-8')
                if re.search(r'\b' + skill + r'\b', sent, re.IGNORECASE):
                    yield CandidateSkill(
                        skill_name=skill,
                        matched_skill=skill,
                        confidence=100,
                        context=sent
                    )
