from collections import defaultdict

class EmbeddingEvaluator(object):
    def __init__(self, ontology, processing_pipeline, metric_list):
        self.ontology = ontology
        self.embedding_model = embedding_model
        self.processing_pipeline = processing_pipeline
        self.metric_list = metric_list
        self.result = None

    def create_metric_objects(self):
        return []

    def evaluate(self):
        result = defaultdict(dict)
        clustering_list = self.ontology.generate_clusterings()
        for clustering in clustering_list:
            for metric in self.metric_objects:
                result[clustering.name][metric.name] = metric.eval(processing_pipeline, clustering)
        self.result = result
        return result
