import json
import logging
import re

import nltk

from descriptors import cachedproperty

from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator
from skills_ml.algorithms.string_cleaners import NLPTransforms


class CandidateSkill(object):
    def __init__(self, skill_name, matched_skill, context, confidence):
        self.skill_name = skill_name
        self.matched_skill = matched_skill
        if isinstance(self.matched_skill, (bytes, bytearray)):
            self.matched_skill = matched_skill.decode('utf-8')
        self.context = context
        self.confidence = confidence


class JobPosting(object):
    def __init__(self, job_posting_json, corpus_creator=None):
        self.job_posting_json = job_posting_json
        self.properties = json.loads(self.job_posting_json.decode('utf-8'))
        if corpus_creator:
            self.corpus_creator = corpus_creator
        else:
            self.corpus_creator = SimpleCorpusCreator()

    @cachedproperty
    def text(self):
        return self.corpus_creator._join(self.properties)

    @cachedproperty
    def id(self):
        return self.properties['id']

    def __getattr__(self, attr):
        return self.properties.get(attr, None)


class ListBasedSkillExtractor(object):
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
