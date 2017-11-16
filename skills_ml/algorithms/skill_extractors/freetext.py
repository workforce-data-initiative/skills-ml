import csv
import logging
from collections import Counter, defaultdict
import smart_open

import nltk
try:
    nltk.sent_tokenize('test')
except LookupError:
    nltk.download('punkt')

from skills_ml.algorithms.string_cleaners import NLPTransforms

from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator

import json
from collections import Counter, OrderedDict, defaultdict
import uuid

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import re


class FreetextSkillExtractor(object):
    """Extract skills from unstructured text

    Originally written by Kwame Porter Robinson
    """
    name = 'onet_ksat_exact'
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
    name = 'onet_ksat_occscoped_exact'
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


class FuzzySkillExtractor(FreetextSkillExtractor):
    name = 'onet_ksat_fuzzy'
    def reg_ex(self, s):
        s = s.replace(".","\.")
        s = s.replace("^","\^")
        s = s.replace("$","\$")
        s = s.replace("*","\*")
        s = s.replace("+","\+")
        s = s.replace("?","\?")
        return s

    def _skills_lookup(self):
        """Create skills lookup

        Reads the object's filename containing skills into a lookup

        Returns: (set) skill names
        """
        with open(self.skills_filename) as infile:
            reader = csv.reader(infile, delimiter='\t')
            header = next(reader)
            index=3
            generator = (self.reg_ex(row[index]) for row in reader)
            
            return set(generator)

    def ie_preprocess(self, document):
        """This function takes raw text and chops and then connects the process to break     
           it down into sentences"""
        
        # Pre-processing
        # e.g.","exempli gratia"
        document=document.replace("e.g.","exempli gratia")
        
        # Sentence tokenizer out of nltk.sent_tokenize
        split = re.split('\n|\*', document)
        
        # Sentence tokenizer
        sentences=[]
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

    def candidate_skills(self, document):
        sentences = self.ie_preprocess(document)
        
        skills = defaultdict(list)
        for skill in available_skills:
            len_skill = len(skill.split())
            for sent in sentences:
                sent = sent.encode('utf-8')
                
                #Exact matching
                if len_skill ==1:
                    sent = sent.decode('utf-8')
                    if re.search(r'\b'+skill +r'\b', sent, re.IGNORECASE):
                        skills[skill] = [100]
                        skills[skill].append(sent)
                #Fuzzy matching
                else:        
                    ratio = fuzz.partial_ratio(skill,sent)
                    # You can adjust the partial of matching here: 100 => exact matching 0 => no matching
                    if ratio > 88:
                        skills[skill] = [ratio]
                        skills[skill].append(sent)
        return skills

class Sample(object):
    def __init__(self, base_path, sample_name):
        self.base_path = base_path
        self.sample_name = sample_name
        self.full_path = '/'.join([self.base_path, sample_name])

    def __iter__(self):
        with smart_open(self.full_path) as f:
            for line in f:
                yield line

def generate_skill_candidates(candidates_path, sample, skill_extractor):
    corpus_creator = SimpleCorpusCreator()
    corpus_creator.document_schema_fields = ['description']
    for job_posting_string in sample:
        job_posting = json.loads(job_posting_string)
        document = corpus_creator._join(job_posting)
        candidate_skills = skill_extractor.candidate_skills(document)
        for candidate_skill, metadata in candidate_skills.items():
            cs_id = str(uuid.uuid4())
            candidate_filename = '/'.join([candidates_path, sample.name, skill_extractor.name, cs_id])
            with open(candidate_filename, 'w') as candidate_file:
                candidate_metadata = {
                    'job_posting_id': job_posting['id'],
                    'candidate_skill': candidate_skill,
                    'confidence': metadata[0],
                    'context': metadata[1],
                    'skill_extractor_name': skill_extractor.name
                }
                json.dump(obj, candidate_file)
        

# TODO: define name attribute on skill extractor

# one invocation:
# get 24k sample into list
# call skill candidate generator with 24k sample and exact match
# 
# call skill candidate generator

if __name__ == '__main__':
    sample_path_config = 'open-skills-private/sampled_jobpostings'
    candidates_path_config = 'open-skills-private/skill_candidates'
    sample = Sample(sample_path_config, 'samples_300_v1')
    skill_extractor = FuzzySkillExtractor("output/skills_master_table.tsv")
    generate_skill_candidates(candidates_path_config, sample, skill_extractor)
