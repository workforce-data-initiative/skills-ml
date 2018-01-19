import json
import logging
import uuid

import boto

from skills_utils.s3 import split_s3_path
from skills_ml.algorithms.corpus_creators.basic import SimpleCorpusCreator

from .base import JobPosting
from .fuzzy_match import FuzzyMatchSkillExtractor
from .exact_match import ExactMatchSkillExtractor
from .soc_exact import SocScopedExactMatchSkillExtractor


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

__all__ = [
    'ExactMatchSkillExtractor',
    'FuzzyMatchSkillExtractor',
    'SocScopedExactMatchSkillExtractor'
]
