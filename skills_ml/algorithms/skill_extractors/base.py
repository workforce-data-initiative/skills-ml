"""Base classes for skill extraction"""
import logging
from abc import ABCMeta, abstractmethod

from collections import Counter


from skills_ml.job_postings.corpora import SimpleCorpusCreator
from skills_ml.algorithms.string_cleaners import NLPTransforms

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


class CandidateSkill(object):
    def __init__(self, skill_name: Text, matched_skill: Text, context: Text, confidence: float):
        """An object holding a text snippet that may be a skill/competency.

        Does not hold all needed contextual metadata about the document (like job posting id).
        This is expected to be managed by the caller.

        Args:
            skill_name (string) The skill found in the text
            matched_skill (string) The matching skill in some reference ontology.
            context (string) The skill_name with surrounding context in the document.
            confidence (float) How sure the skill extractor is that this is a skill (range 0-1)
        """
        self.skill_name = skill_name
        self.matched_skill = matched_skill
        if isinstance(self.matched_skill, (bytes, bytearray)):
            self.matched_skill = matched_skill.decode('utf-8')
        self.context = context
        self.confidence = confidence


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
        self.nlp = NLPTransforms()

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
            skill_counts[self.nlp.lowercase_strip_punc(candidate_skill.skill_name)] += 1
        return skill_counts


class ListBasedSkillExtractor(SkillExtractor):
    """Extract skills by comparing with a known lookup/list


    Subclasses must implement _skills_lookup and _document_skills_in_lookup

    Args:
        skill_lookup_path (string) A path to the skill lookup file
        skill_lookup_name (string, optional) An identifier for the skill lookup type.
            Defaults to onet_ksat
        skill_lookup_description (string, optional) A human-readable description of the skill lookup.
    """
    def __init__(
            self,
            skill_lookup_path,
            skill_lookup_name='onet_ksat',
            skill_lookup_description=None,
            *args,
            **kwargs
    ):
        super(ListBasedSkillExtractor, self).__init__(*args, **kwargs)
        self.skill_lookup_path = skill_lookup_path
        self.skill_lookup_name = skill_lookup_name
        # TODO: get this from competency object when they are in here
        # at that point there should also be no default, make them pass in the object
        if skill_lookup_name == 'onet_ksat':
            self.skill_lookup_description = 'ONET Knowledge, Skills, Abilities, Tools, and Technology'
        else:
            self.skill_lookup_description = skill_lookup_description or ''
        self.lookup = self._skills_lookup()
        logging.info(
            'Done creating skills lookup with %d entries',
            len(self.lookup)
        )
        self.trie_regex = trie_regex_from_words(self.lookup)

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
        return f'{self.skill_lookup_name}_{self.method_name}'

    @property
    def description(self):
        return f'{self.skill_lookup_description} found by {self.method_description}'

    def candidate_skills_in_context(self, context):
        matches = self.trie_regex.findall(context)
        for match in matches: 
            logging.info('Yielding exact match %s in string %s', match, context)
            yield CandidateSkill(
                skill_name=match.lower(),
                matched_skill=match,
                confidence=100,
                context=context
            )

