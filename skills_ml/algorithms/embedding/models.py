"""Embedding model class inherited the interface from gensim"""
from skills_ml.storage import FSStore, ModelStorage
from skills_ml.algorithms.embedding.base import BaseEmbeddingModel

import gensim
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.fasttext import FastText as FT_gensim

from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin

import os
import numpy as np
import logging

class Word2VecModel(BaseEmbeddingModel, Word2Vec):
    """The Word2VecModel inherited from gensim's Word2Vec model (
    https://radimrehurek.com/gensim/models/word2vec.html) for training,
    using and evaluating word embedding with extension methods.

    Example:
    ```
    from skills_ml.algorithms.embedding.models import Word2VecModel

    word2vec_model = Word2VecModel()
    ```
    """
    def __init__(self, model_name=None, storage=None, *args, **kwargs):
        """
        Attributes:
            storage (:obj: `skills_ml.Store`): skills_ml Store object
            model (:obj: `gensim.models.doc2vec.Doc2Vec`): gensim doc2vec model.
        """
        BaseEmbeddingModel.__init__(self, model_name=model_name, storage=storage)
        Word2Vec.__init__(self, *args, **kwargs)
        self.model_type = Word2Vec.__name__.lower()

    def infer_vector(self, doc_words, warning=False):
        """
        Average all the word-vectors together and ignore the unseen words
        Arg:
            doc_words (list): a list of tokenized words
        Returns:
            a vector representing a whole doc/sentence
        """
        sum_vector = np.zeros(self.vector_size)
        words_in_vocab = []
        for token in doc_words:
            try:
                sum_vector += self.wv[token]
                words_in_vocab.append(token)
            except KeyError as e:
                if warning:
                    logging.warning("".join([str(e), ". Ignore the word."]))
                pass

        if len(words_in_vocab) == 0:
            if warning:
                logging.warning("None of the words is in vocabulary.")
            return np.random.rand(self.vector_size)
        sentence_vector = sum_vector / len(words_in_vocab)
        return sentence_vector


class Doc2VecModel(BaseEmbeddingModel, Doc2Vec):
    """The Doc2VecModel inherited from gensim's Doc2Vec model (
    https://radimrehurek.com/gensim/models/doc2vec) for training,
    using and evaluating word embedding with extension methods.

    Example:
    ```
    from skills_ml.algorithms.embedding.models import Doc2VecModel

    doc2vec_model = Doc2VecModel()
    ```
    """
    def __init__(self, model_name=None, storage=None, *args, **kwargs):
        BaseEmbeddingModel.__init__(self, model_name=model_name, storage=storage)
        Doc2Vec.__init__(self, *args, **kwargs)
        self.model_type = Doc2Vec.__name__.lower()
        self.lookup_dict = None

class FastTextModel(BaseEmbeddingModel, FT_gensim):
    """The FastTextModel inhereited from gensim's FastText model (
    https://radimrehurek.com/gensim/models/fasttext.html) for training,
    using and evaluating word embedding with extension methods.

    Example:
        ```
        from skills_ml.algorithms.embedding.models import import FastTextModel

        fasttext = FastTextModel()
        ```
    """
    def __init__(self, model_name=None, storage=None, *args, **kwargs):
        BaseEmbeddingModel.__init__(self, model_name=model_name, storage=storage)
        FT_gensim.__init__(self, *args, **kwargs)
        self.model_type = FT_gensim.__name__.lower()

    def infer_vector(self, doc_words, warning=False):
         """
         Average all the word-vectors together and ignore the unseen words
         """
         sum_vector = np.zeros(self.vector_size)
         words_in_vocab = []
         for token in doc_words:
             try:
                 sum_vector += self.wv[token]
                 words_in_vocab.append(token)
             except KeyError as e:
                 if warning:
                    logging.warning("".join([str(e), ". Ignore the word."]))
                 pass

         if len(words_in_vocab) == 0:
             if warning:
                logging.warning("None of the words is in vocabulary.")
             return np.random.rand(self.vector_size)
         sentence_vector = sum_vector / len(words_in_vocab)
         return sentence_vector


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def fit(self, X, y):
        return self

    def transform(self, X):
        trans_X = []
        for x in X:
            trans_X.append(self.embedding_model.infer_vector(x))
        return trans_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def visualize_in_tensorboard(embedding_model, output_dirname=None, host="127.0.0.1"):
    if output_dirname is None:
        output_dirname =  embedding_model.model_name.split('.')[0]

    meta_file = f"{output_dirname}_metadata.tsv"
    output_path = os.path.join(os.getcwd(), output_dirname)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, meta_file), "wb") as file_metadata:
        for word in embedding_model.wv.index2word:
            file_metadata.write(gensim.utils.to_utf8(word) + gensim.utils.to_utf8("\n"))

    embedding = tf.Variable(embedding_model.wv.vectors, trainable = False, name = f"{output_dirname}_tensor")
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = f"{output_dirname}_tensor"
        embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
        projector.visualize_embeddings(writer, config)
        saver.save(sess, os.path.join(output_path,f"{output_dirname}_metadata.ckpt"))

    print(f"Run `tensorboard --logdir={output_path} --host {host}` to run visualize result on tensorboard")

