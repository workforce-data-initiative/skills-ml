from collections import defaultdict
import logging
import re
import unicodecsv as csv

from smart_open import smart_open

from .base import CandidateSkill, trie_regex_from_words
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

        soc_trie_regex = trie_regex_from_words(self.lookup[soc_code])

        for sent in sentences:
            matches = soc_trie_regex.findall(sent)
            for match in matches: 
                logging.info('Yielding exact match %s in string %s', match, sent)
                yield CandidateSkill(
                    skill_name=match.lower(),
                    matched_skill=match,
                    confidence=100,
                    context=sent
                )
