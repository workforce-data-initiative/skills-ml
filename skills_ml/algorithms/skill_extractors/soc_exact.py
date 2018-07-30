from collections import defaultdict
import logging
import re
import unicodecsv as csv

from smart_open import smart_open

from skills_ml.ontologies.base import CompetencyOntology

from .base import CandidateSkill, trie_regex_from_words
from .exact_match import ExactMatchSkillExtractor


class SocScopedExactMatchSkillExtractor(ExactMatchSkillExtractor):
    """Extract skills from unstructured text,
    but only return matches that agree with a known taxonomy
    """
    method_name = 'occscoped_exact_match'
    method_description = 'Exact matching using only the skills known by ONET to be associated with the given SOC code'

    def __init__(self, competency_ontology, *args, **kwargs):
        if not isinstance(competency_ontology, CompetencyOntology):
            raise ValueError('Must pass in a CompetencyOntology object')
        super().__init__(competency_ontology.competency_framework, *args, **kwargs)

        self.skill_extractor_by_soc = {}
        for occupation in competency_ontology.occupations:
            subontology = competency_ontology.filter_by(
                lambda edge: edge.occupation == occupation,
                competency_name='competencies_' + occupation.identifier,
                competency_description='Competencies needed by ' + occupation.name
            )
            self.skill_extractor_by_soc[occupation.identifier] = ExactMatchSkillExtractor(subontology.competency_framework)

    def candidate_skills(self, source_object):
        soc_code = source_object.get('onet_soc_code', None)
        if not soc_code or soc_code not in self.skill_extractor_by_soc:
            return
        yield from self.skill_extractor_by_soc[soc_code].candidate_skills(source_object)
