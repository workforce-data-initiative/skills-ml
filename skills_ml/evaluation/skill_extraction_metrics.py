import json
import logging
from abc import ABCMeta, abstractmethod
from skills_ml.job_postings.common_schema import get_onet_occupation
from skills_ml.ontologies.base import CompetencyOntology
from skills_ml.algorithms.skill_extractors.base import CandidateSkillYielder
from skills_ml.algorithms.sampling import Sample
from collections import defaultdict
from typing import List
import numpy
import statistics


class SkillExtractorMetric(metaclass=ABCMeta):
    @abstractmethod
    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> float:
        pass


class OntologyCompetencyRecall(SkillExtractorMetric):
    """The percentage of competencies in an ontology which are present in the candidate skills"""

    @property
    def name(self):
        return f'{self.ontology.competency_framework.name}_competency_recall'

    def __init__(self, ontology: CompetencyOntology):
        self.ontology = ontology
        self.lookup = set(competency.identifier for competency in ontology.competencies)

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> float:
        num_total_terms = len(self.lookup)
        if num_total_terms == 0:
            logging.warning('Lookup has zero terms, cannot evaluate. Returning 0')
            return 0
        found_terms = set()
        for candidate_skill in candidate_skills:
            if candidate_skill.matched_skill_identifier in found_terms:
                continue
            if candidate_skill.matched_skill_identifier in self.lookup:
                found_terms.add(candidate_skill.matched_skill_identifier)
        num_found_terms = len(found_terms)
        logging.info('Found %s terms out of %s total', num_found_terms, num_total_terms)
        return float(num_found_terms)/num_total_terms


class OntologyOccupationRecall(SkillExtractorMetric):
    """The percentage of occupations in the ontology that are present in the candidate skills"""

    @property
    def name(self):
        return f'{self.ontology.name}_occupation_recall'

    def __init__(self, ontology: CompetencyOntology):
        self.ontology = ontology
        self.lookup = set(occupation.identifier.lower() for occupation in ontology.occupations)

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> float:
        num_total_occupations = len(self.lookup)
        num_total_terms = len(self.lookup)
        if num_total_terms == 0:
            logging.warning('Lookup has zero terms, cannot evaluate. Returning 0')
            return 0
        found_occupations = set()
        for candidate_skill in candidate_skills:
            occupation = get_onet_occupation(candidate_skill.source_object)
            if occupation and occupation not in found_occupations:
                found_occupations.add(occupation)
        num_found_occupations = len(found_occupations) 
        logging.info('Found %s occupations out of %s total', num_found_occupations, num_total_occupations)
        return float(num_found_occupations)/num_total_occupations


class MedianSkillsPerDocument(SkillExtractorMetric):
    """The median number of distinct skills present in each document"""

    name = 'median_skills_per_document'

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> float:
        skills_in_document = defaultdict(set)
        for candidate_skill in candidate_skills:
            if candidate_skill.skill_name not in skills_in_document[candidate_skill.document_id]:
                skills_in_document[candidate_skill.document_id].add(candidate_skill.skill_name)
        documents_with_skills = len(skills_in_document.values())
        counts = [len(skill_list) for skill_list in skills_in_document.values()]
        for _ in range(0, sample_len - documents_with_skills):
            counts.append(0)
        return statistics.median(counts)


class SkillsPerDocumentHistogram(SkillExtractorMetric):
    """The"""
    def __init__(self, bins=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins

    @property
    def name(self):
        return f'skills_per_document_histogram_{self.bins}bins'

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> List:
        skills_in_document = defaultdict(set)
        for candidate_skill in candidate_skills:
            if candidate_skill.skill_name not in skills_in_document[candidate_skill.document_id]:
                skills_in_document[candidate_skill.document_id].add(candidate_skill.skill_name)
        documents_with_skills = len(skills_in_document.values())
        counts = [len(skill_list) for skill_list in skills_in_document.values()]
        for _ in range(0, sample_len - documents_with_skills):
            counts.append(0)
        return list(numpy.histogram(counts, bins=self.bins)[0])


class PercentageNoSkillDocuments(SkillExtractorMetric):
    """The percentage of documents that contained zero skills"""

    name = 'pct_no_skill_documents'

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> float:
        documents_with_skills = set()
        for candidate_skill in candidate_skills:
            if candidate_skill.document_id not in documents_with_skills:
                documents_with_skills.add(candidate_skill.document_id)

        return (sample_len - len(documents_with_skills)) / sample_len


class TotalVocabularySize(SkillExtractorMetric):
    """The total number of skills represented"""

    name = 'total_vocabulary_size'

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> int:
        skills = set()
        for candidate_skill in candidate_skills:
            if candidate_skill.skill_name not in skills:
                skills.add(candidate_skill.skill_name)
        return len(skills)


class TotalOccurrences(SkillExtractorMetric):
    """The total number of candidate skill occurrences"""

    name = 'total_candidate_skills'

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> int:
        return sum(1 for candidate_skill in candidate_skills)


strict_candidate_key = lambda cs: (cs.document_id, cs.document_type, cs.skill_name, cs.start_index)
nonstrict_candidate_key = lambda cs: (cs.document_id, cs.document_type, cs.skill_name)

class EvaluationSetPrecision(SkillExtractorMetric):
    """Find the precision evaluated against an evaluation set of candidate skills.

    Args:
    candidate_skills (CandidateSkillYielder): A collection of candidate skills to evaluate against
    evaluation_set_name (str): A name for the evaluation set of candidate skills.
        Used in the name of the metric so results from multiple evaluation sets
        can be compared side-by-side.
    strict (bool, default True): Whether or not to enforce the exact location of the match,
        versus just matching between sets on the same skill name and document.
        Setting this to False will guard against:
            1. labelers who don't mark every instance of a skill once they found one instance
            2. discrepancies in start_index values caused by errant transformation methods 
        However, this could also produce false matches, so use with care.
    """

    @property
    def name(self):
        strictstring = 'strict' if self.strict else 'nonstrict'
        return f'{self.evaluation_set_name}_evaluation_set_precision_{strictstring}'

    def __init__(
        self,
        candidate_skills: CandidateSkillYielder,
        evaluation_set_name: str,
        strict: bool=True
    ):
        self.strict = strict
        self.keyfunc = strict_candidate_key if strict else nonstrict_candidate_key
        self.gold_standard_candidate_skills = set(self.keyfunc(candidate_skill) for candidate_skill in candidate_skills)
        self.evaluation_set_name = evaluation_set_name

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> int:
        num_candidates_in_gs = 0
        total_candidates = 0
        for candidate_skill in candidate_skills:
            total_candidates += 1
            if self.keyfunc(candidate_skill) in self.gold_standard_candidate_skills:
                num_candidates_in_gs += 1
        return float(num_candidates_in_gs) / total_candidates


class EvaluationSetRecall(SkillExtractorMetric):
    """Find the recall evaluated against an evaluation set of candidate skills.

    Args:
    candidate_skills (CandidateSkillYielder): A collection of candidate skills to evaluate against
    evaluation_set_name (str): A name for the evaluation set of candidate skills.
        Used in the name of the metric so results from multiple evaluation sets
        can be compared side-by-side.
    strict (bool, default True): Whether or not to enforce the exact location of the match,
        versus just matching between sets on the same skill name and document.
        Setting this to False will guard against:
            1. labelers who don't mark every instance of a skill once they found one instance
            2. discrepancies in start_index values caused by errant transformation methods 
        However, this could also produce false matches, so use with care.
    """

    @property
    def name(self):
        strictstring = 'strict' if self.strict else 'nonstrict'
        return f'{self.evaluation_set_name}_evaluation_set_recall_{strictstring}'

    def __init__(self, candidate_skills, evaluation_set_name, strict=True):
        self.strict = strict
        self.keyfunc = strict_candidate_key if strict else nonstrict_candidate_key
        self.gold_standard_candidate_skills = set(self.keyfunc(candidate_skill) for candidate_skill in candidate_skills)
        self.evaluation_set_name = evaluation_set_name

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> int:
        found_candidate_skills = set(self.keyfunc(candidate_skill) for candidate_skill in candidate_skills)

        num_candidates_found = 0
        for cs in self.gold_standard_candidate_skills:
            if cs in found_candidate_skills:
                num_candidates_found += 1
        return float(num_candidates_found) / len(self.gold_standard_candidate_skills)
