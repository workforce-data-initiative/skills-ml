import csv
import logging
from collections import Counter, defaultdict

from skills_ml.algorithms.string_cleaners import NLPTransforms


class FreetextSkillExtractor(object):
    """Extract skills from unstructured text

    Originally written by Kwame Porter Robinson
    """
    def __init__(self, skills_filename):
        self.skills_filename = skills_filename
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

    def _skills_lookup(self):
        """Create skills lookup

        Reads the object's filename containing skills into a lookup

        Returns: (set) skill names
        """
        logging.info('Creating skills lookup from %s', self.skills_filename)
        with open(self.skills_filename) as infile:
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

    def document_skill_counts(self, document):
        """Count skills in the document

        Args:
            document (string) A document for searching, such as a job posting

        Returns: (collections.Counter) skills found in the document, all
            values set to 1 (multiple occurrences of a skill do not count)
        """
        return self._document_skills_in_lookup(document, self.lookup)


class OccupationScopedSkillExtractor(FreetextSkillExtractor):
    """Extract skills from unstructured text,
    but only return matches that agree with a known taxonomy
    """
    def _skills_lookup(self):
        """Create skills lookup

        Reads the object's filename containing skills into a lookup

        Returns: (set) skill names
        """
        logging.info('Creating skills lookup from %s', self.skills_filename)
        lookup = defaultdict(set)
        with open(self.skills_filename) as infile:
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


class FakeFreetextSkillExtractor(FreetextSkillExtractor):
    """A skill extractor that takes a list of skills
    instead of reading from a filename
    """
    def __init__(self, skills):
        """
        Args:
            skills (list) skill names that the extractor should use
        """
        self.skills = skills
        super(FakeFreetextSkillExtractor, self).__init__('')

    def _skills_lookup(self):
        return set(self.skills)


class FakeOccupationScopedSkillExtractor(OccupationScopedSkillExtractor):
    """A skill extractor that takes a list of skills
    instead of reading from a filename
    """
    def __init__(self, skills):
        """
        Args:
            skills (dict) soc codes as keys, lists of skill names as values
        """
        self.skills = skills
        super(FakeOccupationScopedSkillExtractor, self).__init__('')

    def _skills_lookup(self):
        return self.skills
