from collections import defaultdict
import logging
import re
import unicodecsv as csv

from smart_open import smart_open

from .base import CandidateSkill
from .exact_match import ExactMatchSkillExtractor


class SocScopedExactMatchSkillExtractor(ExactMatchSkillExtractor):
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
