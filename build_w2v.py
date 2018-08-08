from skills_ml.algorithms.embedding.models import Word2VecModel, Doc2VecModel
from skills_ml.algorithms.embedding.train import EmbeddingTrainer
from skills_ml.job_postings.common_schema import JobPostingCollectionFromS3
from skills_ml.job_postings.filtering import JobPostingFilterer
from skills_ml.job_postings.corpora import Word2VecGensimCorpusCreator, Doc2VecGensimCorpusCreator
from skills_ml.algorithms.occupation_classifiers.classifiers import KNNDoc2VecClassifier
import boto
s3_conn = boto.connect_s3()
from skills_ml.storage import S3Store
s3_storage = S3Store('open-skills-private/model_cache/embedding')
import multiprocessing
import logging

num_of_worker = multiprocessing.cpu_count()


def has_soc_filter(document):
    if document['onet_soc_code'] != None and document['onet_soc_code'] != '':
        return True
    else:
        return False

jp = JobPostingCollectionFromS3(s3_conn, ['nlx-postings-common-schema/2015'])

jp_f = JobPostingFilterer(jp, [has_soc_filter])
corpus_generator = Word2VecGensimCorpusCreator(jp_f)
w2v = Word2VecModel(storage=s3_storage, size=300, min_count=3, hs=1,  iter=3, window=8, sample=1e-5, workers=num_of_worker)
trainer = EmbeddingTrainer(corpus_generator, w2v)
trainer.train()
w2v.save()
logging.info(f"word2vec {w2v.model_name} is trained!")
