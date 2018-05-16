from skills_ml.algorithms.skill_feature_creator import FeatureCreator
from skills_ml.algorithms.string_cleaners import NLPTransforms
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_utils.s3 import upload, list_files, download

import gensim
import os
from moto import mock_s3_deprecated
from mock import patch
import boto
import tempfile

import json

import pytest

docs = '{"experienceRequirements": "To apply, candidates must submit a Candidate Profile through Jobs@UVa (https://jobs.virginia.edu), search on posting number 0615449, and electronically attach the following: a cover letter of interest describing teaching experience, preferred semester, and concentration area, a curriculum vitae, and contact information for three letters of reference., Studio Art at the University of Virginia is an undergraduate program with a small post-baccalaureate \\"fifth-year\\" fellowship program. The department offers concentrations in film, new media, painting, photography, printmaking, and sculpture, and artists working in any medium are invited to apply., Please direct questions about the position to William Wylie, Search Committee Chair, ww9b@virginia.edu., Questions regarding the application process in JOBS@UVa should be directed to: Ashley Watkins, adf2p@virginia.edu, 434-243-1785., The University of Virginia is an affirmative action/equal opportunity employer committed to diversity, equity, and inclusiveness. Women, minorities, veterans and persons with disabilities are encouraged to apply., The University will perform background checks on all new faculty hires prior to making a final offer of employment., Employment Conditions for Faculty, CV / ResumeCover LetterContact information for 3 References, name, email, phone", "description": "## Posting Summary:\\n\\nThe McIntire Department of Art at the University of Virginia seeks applications from visual artists of international stature for two positions as the Ruffin Distinguished Artist in Residence for either the spring term of 2016 or the spring term of 2017.\\nEach Ruffin Distinguished Artist in Residence position is a one-semester nonrenewable appointment. The compensation for this appointment will be commensurate with the highest levels of artistic distinction, and includes housing, a studio, and research support. The successful candidate will have an exceptional record of exhibitions and honors both in the U.S. and abroad, and will express a commitment to contribute to the life of the Studio Art Program at the University of Virginia and within the University as a whole. The appointment will begin in January at the start of the spring semester, and the candidate must be a resident in Charlottesville while classes are in session. Duties will include teaching one advanced seminar course and public presentations by agreement with the Chair of the McIntire Department of Art, and the Associate Chair of Studio Art.\\n\\nThe Ruffin Distinguished Artist in Residence is a teaching position. Teaching experience is required and an MFA or equivalent terminal degree is strongly preferred. Preferred Application Deadline: January 15, 2015. The position will remain open until filled.\\n\\nTo apply, candidates must submit a Candidate Profile through Jobs@UVa (https://jobs.virginia.edu), search on posting number 0615449, and electronically attach the following: a cover letter of interest describing teaching experience, preferred semester, and concentration area, a curriculum vitae, and contact information for three letters of reference.\\n\\nStudio Art at the University of Virginia is an undergraduate program with a small post-baccalaureate \\"fifth-year\\" fellowship program. The department offers concentrations in film, new media, painting, photography, printmaking, and sculpture, and artists working in any medium are invited to apply.\\n\\n\\n\\nPlease direct questions about the position to William Wylie, Search Committee Chair, ww9b@virginia.edu.\\n\\nQuestions regarding the application process in JOBS@UVa should be directed to: Ashley Watkins, adf2p@virginia.edu, 434-243-1785.\\n\\nThe University of Virginia is an affirmative action/equal opportunity employer committed to diversity, equity, and inclusiveness. Women, minorities, veterans and persons with disabilities are encouraged to apply.\\n\\n\\nThe University will perform background checks on all new faculty hires prior to making a final offer of employment.\\n\\n\\n\\n\\n\\n## University Leadership Characteristics:\\n\\n*For Thomas Jefferson, learning was an integral part of life. The \\"academical village\\" is based on the assumption that the life of the mind is a pursuit for all participants in the University, that learning is a lifelong and shared process, and that interaction between scholars and students enlivens the pursuit of knowledge.*\\n\\n\\nUniversity Human Resources strives to identify applicants who will contribute as high potential employees, leaders and managers. We employ individuals who foster and promote the University mission and purpose. Successful candidates exemplify Uncommon Integrity; they are honest, trusted, team-oriented and live the core values of the University. These candidates display Great Judgment, by practicing evidence-based decision-making. They are Strategically Focused by contributing to and achieving department goals and vision. They set high performance standards and hold themselves accountable by Aggressively Executing these standards. These employees also develop a Deep Passion for the University and the impact it has on students, faculty, alumni and community. Successful candidates identify their personal career goals and development opportunities. They contribute to team success by Leading Talent through individual efforts.\\n\\n\\n\\n## Employment Conditions for Faculty\\n\\nU.Va. will perform background checks including receipt of official transcripts from the institution granting the highest degree for all new faculty hires prior to making a final offer of employment.\\n\\n\\n\\n## EO/AA Statement:\\n\\nThe University of Virginia is an equal opportunity and affirmative action employer. Women, minorities, veterans and persons with disabilities are encouraged to apply.\\n\\n\\n\\n*Is this position funded in whole or in part by the American Recovery & Reinvestment Act (Stimulus Package)?:* No\\n\\n\\n*Organization (Position Organization):* 31660 AS-Art\\n\\n\\n*Rank:* Lecturer\\n\\n\\n*Preferred EducationWhat level of education is preferred to successfully perform the duties and responsibilities of the position? Choose one.:* No Response\\n\\n\\n*Location:* Charlottesville\\n\\n\\n*Position Type:* Faculty Wage\\n\\n\\n*Required Applicant Documents:* CV / ResumeCover LetterContact information for 3 References - name, email, phone\\n\\n\\n*Department:* Department of Art\\n\\n\\n*Area of Interest:* No Response\\n\\n\\n*Academic Year for Position? (e.g. 2015):* 2016\\n\\n\\n*Employment Posting Category:* Faculty\\n\\n\\n*E-mail a Friend:* jobs.virginia.edu/applicants/Central?quickFind=74914\\n\\n\\n*Working Title:* Ruffin Distinguished Artist in Residence\\n\\n\\n*Appointment Type:* Teaching and Research\\n\\n\\n*Posting Date:* 11-20-2014\\n\\n\\n*Type of Application:(required to apply for this posting):* Candidate Profile\\n\\n\\n*Posting Number:* 0615449\\n\\n\\n*Required EducationWhat is the minimum level of formal education required to successfully perform the duties and responsibilities of the position? Choose one.:* No Response\\n\\n\\n*Closing Date:* Open Until Filled\\n\\n\\n*Posting for UVA Employees Only:* No Response\\n\\n\\n*Tenure Status:* Tenure Ineligible, w/o Expectation of Continued Employment", "id": "VA_8242c6daf049d6cf0fde1f8b92048b28", "baseSalary": {"@type": "MonetaryAmount", "minValue": 0, "maxValue": 0}, "jobLocation": {"@type": "Place", "address": {"addressRegion": "Virginia", "@type": "PostalAddress", "addressLocality": "Charlottesville"}}, "incentiveCompensation": "", "occupationalCategory": "25-1121.00, 27-1013.00, 11-9033.00, 25-1121.00, 27-1013.00, 11-9033.00, 25-1121.00, 27-1013.00, 11-9033.00, 25-1121.00, 27-1013.00, 11-9033.00", "employmentType": "", "datePosted": "2016-03-17", "industry": "", "onet_soc_code": "", "validThrough": null, "skills": "", "@context": "http://schema.org", "title": "Ruffin Distinguished Artist in Residence", "@type": "JobPosting", "qualifications": "", "educationRequirements": ""}\n'

def get_corpus(num):
    lines = [docs]*num
    for line in lines:
        yield json.loads(line)

class FakeCorpusGenerator(object):
    def __init__(self , num=5):
        self.num = num
        self.lookup = {}
        self.nlp = NLPTransforms()

    @property
    def metadata(self):
        meta_dict = {'corpus_creator': ".".join([self.__module__ , self.__class__.__name__])}
        return meta_dict

    def __iter__(self):
        k = 1
        corpus_memory_friendly = get_corpus(num=100)
        for data in corpus_memory_friendly:
            data = data['description']
            data = gensim.utils.to_unicode(data).split(',')
            words = data[0].split()
            label = [str(k)]
            self.lookup[str(k)] = data[2]
            yield gensim.models.doc2vec.TaggedDocument(words, label)
            k += 1

@mock_s3_deprecated
@pytest.mark.skip('Gensim/boto versioning needs figuring out')
def test_skill_feature_creator():
    s3_conn = boto.connect_s3()

    bucket_name = 'fake-bucket'
    bucket = s3_conn.create_bucket(bucket_name)

    s3_prefix = 'fake-bucket/model_cache/embedding/'

    fake_corpus_train = FakeCorpusGenerator(num=10)

    trainer = EmbeddingTrainer(corpus_generator=fake_corpus_train, s3_conn=s3_conn, model_s3_path=s3_prefix, model_type='doc2vec')
    trainer.train()
    model_name = trainer.modelname


    docs = ["example 1", "example 2"]

    fc = FeatureCreator(
        s3_conn,
        features=['StructuralFeature'],
        embedding_model_name=model_name,
        embedding_model_path=s3_prefix
    )
    structural_feature = fc.featurize(docs).__next__()
    assert len(structural_feature) == 1
    assert fc.params['embedding_model_name'] == model_name
    assert fc.params['embedding_model_path'] == s3_prefix

    fc = FeatureCreator(
        s3_conn,
        features=['ContextualFeature'],
        embedding_model_name=model_name,
        embedding_model_path=s3_prefix
    )
    contextual_feature = fc.featurize(docs).__next__()
    assert len(contextual_feature) == 1

    fc = FeatureCreator(
        s3_conn,
        features=['EmbeddingFeature'],
        embedding_model_name=model_name,
        embedding_model_path=s3_prefix
    )

    embedding_feature = fc.featurize(docs).__next__()
    assert len(embedding_feature) == trainer._model.infer_vector(docs[0]).shape[0]
    assert len([f for f in fc.featurize(docs)]) == len(docs)

    fc = FeatureCreator(
        s3_conn,
        features=['StructuralFeature', 'ContextualFeature'],
        embedding_model_name=model_name,
        embedding_model_path=s3_prefix
    )
    assert len(fc.featurize(docs).__next__()) == len(structural_feature + contextual_feature)

    fc = FeatureCreator(
        s3_conn,
        features="all",
        embedding_model_name=model_name,
        embedding_model_path=s3_prefix
    )
    assert len(fc.featurize(docs).__next__()) == len(structural_feature + contextual_feature + embedding_feature)
