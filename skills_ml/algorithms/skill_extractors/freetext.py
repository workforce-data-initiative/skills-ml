import unicodecsv as csv
import logging
from collections import Counter, defaultdict
from smart_open import smart_open
import re

import nltk
try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

from fuzzywuzzy import fuzz

from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.skill_extractors import CandidateSkill


class UnstructuredTextSkillExtractor(object):
    def __init__(self, skill_lookup_path, skill_lookup_type='onet_ksat'):
        self.skill_lookup_path = skill_lookup_path
        self.skill_lookup_type = skill_lookup_type
        self.tracker = {
            'total_skills': 0,
            'jobs_with_skills': 0
        }
        self.nlp = NLPTransforms()
        self.lookup = self._skills_lookup()
        logging.info(
            'Done creating skills lookup with %d entries',
            len(self.lookup)
        )

    def document_skill_counts(self, document):
        """Count skills in the document

        Args:
            document (string) A document for searching, such as a job posting

        Returns: (collections.Counter) skills found in the document, all
            values set to 1 (multiple occurrences of a skill do not count)
        """
        return self._document_skills_in_lookup(document, self.lookup)

    def ie_preprocess(self, document):
        """This function takes raw text and chops and then connects the process to break
           it down into sentences"""

        # Pre-processing
        # e.g.","exempli gratia"
        document = document.replace("e.g.", "exempli gratia")

        # Sentence tokenizer out of nltk.sent_tokenize
        split = re.split('\n|\*', document)

        # Sentence tokenizer
        sentences = []
        for sent in split:
            sents = nltk.sent_tokenize(sent)
            length = len(sents)
            if length == 0:
                next
            elif length == 1:
                sentences.append(sents[0])
            else:
                for i in range(length):
                    sentences.append(sents[i])
        return sentences


class ExactMatchSkillExtractor(UnstructuredTextSkillExtractor):
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
            lookup (object) A collection that can be queried for an individual skill, implementing 'in' (i.e. 'skill in lookup')
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


class OccupationScopedSkillExtractor(ExactMatchSkillExtractor):
    """Extract skills from unstructured text,
    but only return matches that agree with a known taxonomy
    """
    name = 'occscoped_exact'

    def _skills_lookup(self):
        """Create skills lookup

        Reads the object's filename containing skills into a lookup

        Returns: (set) skill names
        """
        logging.info('Creating skills lookup from %s', self.skill_lookup_path)
        lookup = defaultdict(set)
        with smart_open(self.skill_lookup_path) as infile:
            reader = csv.reader(infile, delimiter='\t')
            header = next(reader)
            ksa_index = header.index(self.nlp.transforms[0])
            soc_index = header.index('O*NET-SOC Code')
            for row in reader:
                lookup[row[soc_index]].add(row[ksa_index])
            return lookup

    def document_skill_counts(self, soc_code, document):
        """Count skills in the document

        Args:
            soc_code (string) A trusted SOC code for the job posting
            document (string) A document for searching, such as a job posting

        Returns: (collections.Counter) skills found in the document, that match
            a known set of skills for the SOC code.
            All values set to 1 (multiple occurrences of a skill do not count)
        """
        return self._document_skills_in_lookup(document, self.lookup[soc_code])

    def candidate_skills(self, job_posting):
        document = job_posting.text
        sentences = self.ie_preprocess(document)

        soc_code = job_posting.onet_soc_code
        if not soc_code or soc_code not in self.lookup:
            return

        for skill in self.lookup[soc_code]:
            len_skill = len(skill.split())
            for sent in sentences:
                sent = sent.encode('utf-8')

                # Exact matching
                if len_skill == 1:
                    sent = sent.decode('utf-8')
                    if re.search(r'\b' + skill + r'\b', sent, re.IGNORECASE):
                        yield CandidateSkill(
                            skill_name=skill,
                            matched_skill=skill,
                            confidence=100,
                            context=sent
                        )


class FuzzyMatchSkillExtractor(UnstructuredTextSkillExtractor):
    name = 'fuzzy'

    def reg_ex(self, s):
        s = s.replace(".", "\.")
        s = s.replace("^", "\^")
        s = s.replace("$", "\$")
        s = s.replace("*", "\*")
        s = s.replace("+", "\+")
        s = s.replace("?", "\?")
        return s

    def _skills_lookup(self):
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

    def ngrams(self, sent, n):
        sent_input = sent.split()
        output = []
        for i in range(len(sent_input)-n+1):
            output.append(sent_input[i:i+n])
        return output

    def fuzzy_matches_in_sentence(self, skill, sentence):
        N = len(skill.split())
        doc = self.ngrams(sentence, N)
        doc_join = [b" ".join(d) for d in doc]

        for dj in doc_join:
            ratio = fuzz.partial_ratio(skill, dj)
            if ratio > 88:
                yield CandidateSkill(
                    skill_name=skill,
                    matched_skill=dj,
                    confidence=ratio,
                    context=sentence.decode('utf-8')
                )

    def candidate_skills(self, job_posting):
        document = job_posting.text
        sentences = self.ie_preprocess(document)

        for skill in self.lookup:
            len_skill = len(skill.split())
            for sent in sentences:
                sent = sent.encode('utf-8')

                # Exact matching
                if len_skill == 1:
                    sent = sent.decode('utf-8')
                    if re.search(r'\b' + skill + r'\b', sent, re.IGNORECASE):
                        yield CandidateSkill(
                            skill_name=skill,
                            matched_skill=skill,
                            confidence=100,
                            context=sent
                        )
                # Fuzzy matching
                else:
                    ratio = fuzz.partial_ratio(skill, sent)
                    # You can adjust the partial of matching here:
                    # 100 => exact matching 0 => no matching
                    if ratio > 88:
                        for match in self.fuzzy_matches_in_sentence(skill, sent):
                            yield match
