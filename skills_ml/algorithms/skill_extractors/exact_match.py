"""Use exact matching with a source list to find skills"""

import logging

import nltk
try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

from .base import (
    CandidateSkill,
    ListBasedSkillExtractor,
    CandidateSkillYielder,
    trie_regex_from_words
)

from typing import Dict


class ExactMatchSkillExtractor(ListBasedSkillExtractor):
    """Extract skills from unstructured text

    Builds a lookup based on the 'name' attribute of all competencies in the given framework

    Originally written by Kwame Porter Robinson
    """
    method_name = 'exact_match'
    method_description = 'Exact matching'

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
        self.lookup_regex = trie_regex_from_words(set(self.id_lookup.keys()))

    def _skills_lookup(self) -> set:
        """Create skills lookup

        Read names from Ontology into a set

        Returns: (set) skill names
        """

    def candidate_skills(self, source_object: Dict) -> CandidateSkillYielder:
        """Yield objects which may represent skills from the given source object.

        Looks for exact matches between the reference skill lookup and the object's text.

        Args: source_object (dict) A structured document for searching, such as a job posting

        Yields: CandidateSkill objects
        """
        document = self.transform_func(source_object)
        sentences = self.nlp.sentence_tokenize(document)
        sentence_start = 0
        for sent in sentences:
            matches = self.lookup_regex.finditer(sent)
            for match in matches: 
                logging.info('Yielding exact match %s in string %s', match, sent)
                yield CandidateSkill(
                    skill_name=match[0].lower(),
                    matched_skill_identifier=self.id_lookup[match[0].lower()],
                    confidence=100,
                    context=sent,
                    start_index=sentence_start + match.start(),
                    document_id=source_object['id'],
                    document_type=source_object['@type'],
                    source_object=source_object,
                    skill_extractor_name=self.name
                )
            sentence_start += len(sent)
