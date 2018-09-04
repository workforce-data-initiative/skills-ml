"""Use fuzzy matching with a source list to extract skills
from unstructured text"""
import logging
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
    max_distance = 4
    max_ngrams = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        competencies = set(
            competency
            for competency in self.competency_framework.values()
            if competency.name
        )

        self.id_lookup = dict((competency.name.lower(), competency.identifier) for competency in competencies)

        logging.info(
            'Found %s entries for lookup',
            len(competencies)
        )
        self.symspell = SymSpell(max_dictionary_edit_distance=4)
        self.symspell.create_dictionary(list(self.id_lookup.keys()))

    @property
    def method_name(self) -> Text:
        return f'fuzzy_thresh{self.match_threshold}_maxdist{self.max_distance}_maxngram{self.max_ngrams}'

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

    def candidate_skills(self, source_object: Dict) -> CandidateSkillYielder:
        document = self.transform_func(source_object)
        sentences = self.nlp.sentence_tokenize(document)
        phrase_start = 0

        for sent in sentences:
            for phrase in self.ngrams(sent, self.max_ngrams):
                length_of_phrase = len(phrase)
                phrase_start += length_of_phrase
                max_distance = length_of_phrase - \
                    ceil(length_of_phrase * self.match_threshold/100)
                if max_distance > self.max_distance:
                    max_distance = self.max_distance
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
                        matched_skill_identifier=self.id_lookup[match.term],
                        confidence=100*(length_of_phrase-match.distance)/length_of_phrase,
                        context=sent,
                        start_index=phrase_start,
                        document_id=source_object['id'],
                        document_type=source_object['@type'],
                        source_object=source_object,
                        skill_extractor_name=self.name
                    )
