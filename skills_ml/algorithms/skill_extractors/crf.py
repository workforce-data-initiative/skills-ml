from itertools import groupby
from typing import List, Text
from skills_ml.evaluation.annotators import AnnotationType
from skills_ml.storage import Store
from skills_ml.algorithms.sampling import Sample
from skills_ml.job_postings import JobPosting
import nltk
import re


class CrfTransformer(object):
    WORD_TOKENIZATION_REGEX = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')

    def __init__(
        self,
        sample_base_path: Text,
        storage_engine: Store,
    ):
        self.sample_base_path = sample_base_path
        self.storage_engine = storage_engine

    def sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def word_tokenize(self, text):
        return [t for t in self.WORD_TOKENIZATION_REGEX.split(text) if t]

    def transform_annotations(self, annotations: List[AnnotationType]):
        """
        Transform annotations into sequence tag format and save per job posting and tagger.

        Adapted from https://github.com/nlplab/brat/blob/master/tools/anntoconll.py
        """
        job_postings = {}
        for annotation in annotations:
            if annotation['job_posting_id'] not in job_postings:
                sample = Sample(
                    base_path=self.sample_base_path,
                    sample_name=annotation['sample_name']
                )
                for line in sample:
                    posting = JobPosting(line)
                    job_postings[posting.id] = posting.text
        for key, group_annotations in groupby(annotations, key=lambda a: (a['job_posting_id'], a['tagger_id'])):
            text = job_postings[key[0]]
            sentences = self.sentence_tokenize(text)
            offset = 0
            lines = []
            nonspace_token_seen = False
            for s in sentences:
                tokens = self.word_tokenize(s)
                print(tokens)

                for t in tokens:
                    if not t.isspace():
                        lines.append(['O', offset, offset+len(t), t])
                        nonspace_token_seen = True
                    offset += len(t)
                if nonspace_token_seen:
                    lines.append([])
            lines.pop()

            offset_label = {}
            newlines = []
            for annotation in sorted(group_annotations, key=lambda a: a['start_index']):
                for i in range(annotation['start_index'], annotation['end_index']):
                    if i in offset_label:
                        print('Warning: overlapping annotations')
                    offset_label[i] = annotation

            prev_label = None
            for i, l in enumerate(lines):
                if not l:
                    prev_label = None
                    newlines.append([])
                    continue
                tag, start, end, token = l

                # TODO: warn for multiple, detailed info for non-initial
                label = None
                for o in range(start, end):
                    if o in offset_label:
                        if o != start:
                            print('Warning: annotation-token boundary mismatch')
                        label = offset_label[o]['entity'].upper()
                        break

                if label is not None:
                    if label == prev_label:
                        tag = 'I-'+label
                    else:
                        tag = 'B-'+label
                prev_label = label

                newlines.append([token, tag])
            self.storage_engine.write(
                fname='{}-{}.txt'.format(key[0], key[1]),
                bytes_obj=('\n'.join(' '.join(line) for line in newlines)).encode('utf-8')
            )
