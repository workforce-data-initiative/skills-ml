"""Use fuzzy matching with a source list to extract skills
from unstructured text"""
import unicodecsv as csv
import logging
from smart_open import smart_open
from descriptors import cachedproperty
from math import ceil

import nltk
try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

from .base import (
    CandidateSkill,
    ListBasedSkillExtractor,
    CandidateSkillYielder
)
from typing import Dict, Text, Generator
from .symspell import SymSpell


class FuzzyMatchSkillExtractor(ListBasedSkillExtractor):
    """Extract skills from unstructured text using fuzzy matching"""

    match_threshold = 88

    @property
    def method_name(self) -> Text:
        return f'fuzzy_{self.match_threshold}'

    @property
    def method_description(self) -> Text:
        return f'Fuzzy matching using ratio of most similar substring, ' + \
            f'with a minimum cutoff of {self.match_threshold} percent match'

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

    def ngrams(self, sent: Text, N: int) -> Generator[Text, None, None]:
        """Yield ngrams from sentence

        Args:
            sent (string) A sentence
            N (int) The maximum N-gram length to create. 

        Yields: (string) n-grams in sentence from n=1 up to the given N
        """
        sent_input = self.nlp.word_tokenize(sent)
        for n in range(1, N):
            for i in range(len(sent_input)-n+1):
                yield " ".join(sent_input[i:i+n]).lower()

    @cachedproperty
    def symspell(self):
        """A SymSpell lookup based on the object's list of skills"""
        ss = SymSpell(max_dictionary_edit_distance=4)
        ss.create_dictionary(list(self.lookup))
        return ss

    def candidate_skills(self, source_object: Dict) -> CandidateSkillYielder:
        document = self.transform_func(source_object)
        sentences = self.nlp.sentence_tokenize(document)

        for sent in sentences:
            for phrase in self.ngrams(sent, 5):
                length_of_phrase = len(phrase)
                max_distance = length_of_phrase - \
                    ceil(length_of_phrase * self.match_threshold/100)
                if max_distance > 4:
                    max_distance = 4
                matches = self.symspell.lookup(phrase, 2, max_distance)
                for match in matches:
                    logging.info(
                        'Fuzzy match found: %s corrected to %s in %s',
                        phrase,
                        match.term,
                        sent
                    )
                    yield CandidateSkill(
                        skill_name=phrase,
                        matched_skill=match.term,
                        confidence=100*(length_of_phrase-match.distance)/length_of_phrase,
                        context=sent
                    )
