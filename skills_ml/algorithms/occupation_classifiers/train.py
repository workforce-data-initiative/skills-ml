from gensim.models.doc2vec import Doc2Vec

from skills_ml.datasets import job_postings
from skills_ml.algorithms.corpus_creators.basic import Doc2VecGensimCorpusCreator

from skills_utils.s3 import upload

from datetime import datetime
from itertools import chain
from glob import glob

import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_representation(s3_conn, quarters, jb_s3_path):
    generators = []
    for quarter in quarters:
        generators.append(job_postings(s3_conn, quarter, jb_s3_path))
    job_postings_generator = chain(*generators)
    corpus = Doc2VecGensimCorpusCreator(list(job_postings_generator))
    corpus_list = list(corpus)
    model = Doc2Vec(size=500, min_count=3, iter=10, window=6, workers=2)
    model.build_vocab(corpus_list)
    model.train(corpus_list, total_examples=model.corpus_count, epochs=model.iter)

    training_time = datetime.today().isoformat()
    modelname = 'doc2vec_' + training_time
    model_path = 'tmp/' + modelname + '.model'
    model.save(model_path)

    lookupname = 'doc2vec_lookup_' + training_time
    lookup_path = 'tmp/' + lookupname + '.json'
    with open(lookup_path, 'w') as handle:
        json.dump(corpus.lookup, handle)

    for f in glob('tmp/*{}*'.format(training_time)):
        upload(s3_conn, f, 'open-skills-private/model_cache/' + modelname)


