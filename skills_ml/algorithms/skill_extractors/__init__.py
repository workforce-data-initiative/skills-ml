import json
import logging
import uuid

import boto

from descriptors import cachedproperty
from skills_utils.s3 import split_s3_path
from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator


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


def upload_candidates_from_job_posting_json(
    candidates_path,
    skill_extractor,
    job_posting_json,
    sample_name
):
    corpus_creator = SimpleCorpusCreator()
    corpus_creator.document_schema_fields = ['description']
    job_posting = JobPosting(job_posting_json, corpus_creator)
    logging.info('Extracting skills for job posting %s', job_posting.id)
    candidate_skills = skill_extractor.candidate_skills(job_posting)
    logging.info('Found skills: %s', candidate_skills)
    for candidate_skill in candidate_skills:
        cs_id = str(uuid.uuid4())
        bucket_name, prefix = split_s3_path(candidates_path)
        s3_conn = boto.connect_s3()
        bucket = s3_conn.get_bucket(bucket_name)
        candidate_filename = '/'.join([
            prefix,
            sample_name,
            skill_extractor.name,
            skill_extractor.skill_lookup_type,
            cs_id + '.json'
        ])
        key = boto.s3.key.Key(bucket=bucket, name=candidate_filename)
        key.set_contents_from_string(json.dumps({
            'job_posting_id': job_posting.id,
            'job_posting_properties': job_posting.properties,
            'key': 'description',
            'candidate_skill': candidate_skill.skill_name,
            'matched_skill': candidate_skill.matched_skill,
            'confidence': candidate_skill.confidence,
            'context': candidate_skill.context,
            'skill_extractor_name': skill_extractor.name,
            'skill_type': skill_extractor.skill_lookup_type
        }))
    logging.info('Saved skills')
