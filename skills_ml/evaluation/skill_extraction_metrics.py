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
        self.lookup = set(competency.name.lower() for competency in ontology.competencies)

    def eval(self, candidate_skills: CandidateSkillYielder, sample_len: int) -> float:
        num_total_terms = len(self.lookup)
        if num_total_terms == 0:
            logging.warning('Lookup has zero terms, cannot evaluate. Returning 0')
            return 0
        found_terms = set()
        for candidate_skill in candidate_skills:
            if candidate_skill.matched_skill in found_terms:
                continue
            if candidate_skill.matched_skill in self.lookup:
                found_terms.add(candidate_skill.matched_skill)
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
