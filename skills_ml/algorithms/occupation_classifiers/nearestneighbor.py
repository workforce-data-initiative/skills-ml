from gensim.models import Doc2Vec
from skills_ml.algorithms.occupation_classifiers import base
import numpy as np

class NearestNeighbors(base.VectorModel):
    def __init__(self, **kwargs):
        super(NearestNeighbors, self).__init__(**kwargs)
        self.training_data = self.model.docvecs.doctag_syn0
        self.target = self._create_target_data()


    def _create_target_data(self):
        y = []
        for i in range(len(self.training_data)):
            y.append(self.lookup[self.model.docvecs.index_to_doctag(i)])

        return np.array(y)


    def classify(self, jobposting, mode='top'):
        """The method to predict the soc code a job posting belongs to.

        Args:
            jobposting (str): a string of cleaned, lower-cased and pre-processed job description context.
            mode (str): a flag of which method to use for classifying.

        Returns:
            tuple(str, float): The predicted soc code and cosine similarity .
        """
        inferred_vector = self.model.infer_vector(jobposting.split())
        if mode == 'top':
            sims = self.model.docvecs.most_similar([inferred_vector], topn=1, indexer=self.indexer)
            resultlist = list(map(lambda l: (self.lookup[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
            predicted_soc = resultlist[0]
            return predicted_soc

        if mode == 'common':
            sims = self.model.docvecs.most_similar([inferred_vector], topn=10, indexer=self.indexer)
            resultlist = list(map(lambda l: (self.lookup[l[0]], l[1]), [(x[0], x[1]) for x in sims]))
            most_common = Counter([r[0] for r in resultlist]).most_common()[0]
            resultdict = defaultdict(list)
            for k, v in resultlist:
                resultdict[k].append(v)

            predicted_soc = (most_common[0], sum(resultdict[most_common[0]])/most_common[1])
            return predicted_soc

