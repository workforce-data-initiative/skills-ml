import pandas as pd

from datasets import negative_dict

from utils.nlp import NLPTransforms


def clean_by_rules():
    pass

def clean_by_neg_dic():
    pass

class JobTitleStringClean(object):
    """
    Clean job titles
    """

    def __init__(self):
        self.negative_dict = negative_dict()

    def clean(self, jobtitles):

        return jobtitles



