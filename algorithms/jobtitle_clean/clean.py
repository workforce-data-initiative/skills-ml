import pandas as pd
import re
from collections import OrderedDict

from datasets import negative_dict

from utils.nlp import NLPTransforms

def clean_by_rules(jobtitle):
    """
    Remove numbers
    :params string jobtitle: A job title string
    :return: A cleaned version of job title
    :rtype: string
    """
    # remove numbers and word with number
    jobtitle = re.sub('[0-9].*', ' ', jobtitle).strip()

    # make one space between words
    jobtitle = ' '.join(jobtitle.split())

    return jobtitle

def clean_by_neg_dic(jobtitle, negative_list):
    """
    Remove words from the negaive dictionary
    :params string jobtitle: A job title string
    :return: A cleaned version of job title
    :rtype: string
    """
    # Exact matching
    result = [word for word in jobtitle.split() if word not in negative_list]
    result2str = ' '.join(result)

    return result2str

class JobTitleStringClean(object):
    """
    Clean job titles
    """

    def __init__(self):
        self.negative_dict = negative_dict()
        self.negative_list = self.negative_dict['places'] + self.negative_dict['states']
    def clean(self, df_jobtitles):
        """
        Clean the job titles by rules and negative dictionary.
        Args:
            df_jobtitles in pandas DataFrame
        Returns:
            pd.DataFrame(cleaned_jobtitles), a clenaed verison of job title in pandas DataFrame

        """
        # Drop the rows with missing vlaue (NaN)
        df_jobtitles = df_jobtitles.dropna()

        columns = list(df_jobtitles.columns)
        cleaned_jobtitles = OrderedDict({key: [] for key in columns})
        for i, row in enumerate(df_jobtitles.values):
            try:
                for colname in columns:
                    if colname == 'title':
                        new_title = clean_by_rules(row[columns.index(colname)])
                        new_title = clean_by_neg_dic(new_title, self.negative_list)
                        cleaned_jobtitles[colname].append(new_title)
                    else:
                        cleaned_jobtitles[colname].append(row[columns.index(colname)])
            except TypeError:
                print('There is some TypeError',row)

        return pd.DataFrame(cleaned_jobtitles)



