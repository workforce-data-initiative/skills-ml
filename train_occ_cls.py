from skills_ml.algorithms.embedding.models import Doc2VecModel
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.job_postings.common_schema import JobPostingCollectionFromS3
from skills_ml.job_postings.filtering import JobPostingFilterer
from skills_ml.job_postings.corpora.basic import Doc2VecGensimCorpusCreator
from skills_ml.algorithms.occupation_classifiers.classifiers import KNNDoc2VecClassifier
from airflow.hooks import S3Hook
s3_conn = S3Hook().get_conn()

from skills_ml.storage import S3Store
s3_storage = S3Store('open-skills-private/model_cache/embedding')

def has_soc_filter(document):
    if document['onet_soc_code'] != None and document['onet_soc_code'] != '':
        return True
    else:
        return False

jp = JobPostingCollectionFromS3(s3_conn, 'open-skills-private/job_postings_common/2011Q3')

jp_f = JobPostingFilterer(jp, [has_soc_filter])
corpus_generator = Doc2VecGensimCorpusCreator(jp_f)
d2v = Doc2VecModel(storage=s3_storage, size=500, min_count=3, iter=4, window=6, workers=3)
trainer = EmbeddingTrainer(corpus_generator, d2v)
trainer.train(lookup=True)
d2v.save()

