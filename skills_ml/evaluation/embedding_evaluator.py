from collections import defaultdict

class EmbeddingEvaluator(object):
    def __init__(self, vectorizing_pipeline, metric_objects_list):
        self.embedding_model = embedding_model
        self.vectorizing_pipeline = vectorizing_pipeline
        self.metric_object_list = metric__object_list
        self.result = None

    def evaluate(self):
        result = defaultdict(dict)
        for metric in self.metric_object_list:
            result[metric.name] = metric.eval(vectorizing_pipeline)
        self.result = result
        return result
