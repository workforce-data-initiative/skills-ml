"""Base classes for skill extraction"""
import logging
from abc import ABCMeta, abstractmethod

from collections import Counter, namedtuple


from skills_ml.job_postings.corpora import SimpleCorpusCreator
from skills_ml.algorithms import nlp
from skills_ml.ontologies.base import CompetencyFramework

from typing import Dict, Callable, Text, Generator
import re


class Trie():
    """Regex::Trie in Python. Creates a Trie out of a list of words. The trie can be exported to a Regex pattern.
    The corresponding Regex should match much faster than a simple Regex union."""

    def __init__(self):
        self.data = {}

    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[''] = 1

    def dump(self):
        return self.data

    def quote(self, char):
        return re.escape(char)

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self):
        return self._pattern(self.dump())


def trie_regex_from_words(words):
    trie = Trie()
    for word in words:
        trie.add(word)
    return re.compile(r"\b" + trie.pattern() + r"\b", re.IGNORECASE)


CandidateSkill = namedtuple('CandidateSkill', [
    'skill_name',
    'matched_skill_identifier',
    'context',
    'start_index',
    'confidence',
    'document_id',
    'document_type',
    'source_object',
    'skill_extractor_name'
])


CandidateSkillYielder = Generator[CandidateSkill, None, None]


class SkillExtractor(object, metaclass=ABCMeta):
    """Abstract class for all skill extractors.

    All subclasses must implement candidate_skills.

    All subclasses must define properties
    'method' (a short machine readable property)
    'description' (a text description of how the extractor does its work)

    Args:
        transform_func (callable, optional) Function that transforms a structured object into text
            Defaults to SimpleCorpusCreator's _join, which takes common text fields
            in common schema job postings and concatenates them together.
            For non-job postings another transform function may be needed.
    """
    def __init__(self, transform_func: Callable=None):
        self.transform_func = transform_func
        if not self.transform_func:
            self.transform_func = SimpleCorpusCreator()._join
        self.nlp = nlp

    @property
    @abstractmethod
    def name(self):
        """A short, machine-friendly (ideally snake_case) name for the skill extractor"""
        pass

    @property
    @abstractmethod
    def description(self):
        """A human-readable description for the skill extractor"""
        pass

    @abstractmethod
    def candidate_skills(self, source_object: Dict) -> CandidateSkillYielder:
        """Yield objects which may represent skills/competencies from the given source object

        Args: source_object (dict) A structured document for searching, such as a job posting

        Yields: CandidateSkill objects
        """
        pass

    def document_skill_counts(self, source_object: Dict):
        """Count skills in the document

        Args:
            source_object (dict) A structured document for searching, such as a job posting

        Returns: (collections.Counter) skills found in the document, all
            values set to 1 (multiple occurrences of a skill do not count)
        """
        skill_counts = Counter()
        for candidate_skill in self.candidate_skills(source_object):
            skill_counts[self.nlp.lowercase_strip_punc(candidate_skill.skill_name).lstrip().rstrip()] += 1
        return skill_counts


class ListBasedSkillExtractor(SkillExtractor):
    """Extract skills by comparing with a known lookup/list.

    Subclasses must implement _skills_lookup and _document_skills_in_lookup

    Args:
        skill_lookup_name (string, optional) An identifier for the skill lookup type.
            Defaults to onet_ksat
        skill_lookup_description (string, optional) A human-readable description of the skill lookup.
    """
    def __init__(self, competency_framework, *args, **kwargs):
        super(ListBasedSkillExtractor, self).__init__(*args, **kwargs)
        if not isinstance(competency_framework, CompetencyFramework):
            raise ValueError('Must pass in a CompetencyFramework object')
        if not competency_framework.name or not competency_framework.description:
            raise ValueError('CompetencyFramework object must be documented with a name and description')
        self.competency_framework = competency_framework

    @property
    @abstractmethod
    def method_name(self):
        """A short, machine-friendly name for the method of skill extraction, independent of the skill lookup"""
        pass

    @property
    @abstractmethod
    def method_description(self):
        """A human readable description for the method of skill extraction, independent of the skill lookup"""
        pass

    @property
    def name(self):
        return f'{self.competency_framework.name}_{self.method_name}'

    @property
    def description(self):
        return f'{self.competency_framework.description} found by {self.method_description}'
