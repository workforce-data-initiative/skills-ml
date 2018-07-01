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
    method_name = 'occscoped_exact_match'
    method_description = 'Exact matching using only the skills known by ONET to be associated with the given SOC code'

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
            ksa_index = header.index('ONET KSA')
            soc_index = header.index('O*NET-SOC Code')
            for row in reader:
                lookup[row[soc_index]].add(row[ksa_index])
            return lookup

    def candidate_skills(self, source_object):
        document = self.transform_func(source_object)
        sentences = self.nlp.sentence_tokenize(document)

        soc_code = source_object.get('onet_soc_code', None)
        if not soc_code or soc_code not in self.lookup:
            return

        for skill in self.lookup[soc_code]:
            for sent in sentences:
                sent = sent.encode('utf-8')

                # Exact matching
                sent = sent.decode('utf-8')
                #print(sent)
                #print(skill)
                if re.search(r'\b' + re.escape(skill) + r'\b', sent, re.IGNORECASE):
                    yield CandidateSkill(
                        skill_name=skill,
                        matched_skill=skill,
                        confidence=100,
                        context=sent
                    )
