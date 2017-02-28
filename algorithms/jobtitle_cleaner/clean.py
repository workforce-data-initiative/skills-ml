import pandas as pd
import re
from collections import OrderedDict

from datasets import negative_positive_dict

def clean_by_rules(jobtitle):
    """
    Remove numbers
    :params string jobtitle: A job title string
    :return: A cleaned version of job title
    :rtype: string
    """
    # remove any words with number in it
    jobtitle = re.sub('\w*\d\w*', ' ', jobtitle).strip()

    # make one space between words
    jobtitle = ' '.join(jobtitle.split())

    return jobtitle

def clean_by_neg_dic(jobtitle, negative_list, positive_list):
    """
    Remove words from the negative dictionary
    :params string jobtitle: A job title string
    :return: A cleaned version of job title
    :rtype: string
    """
    # Exact matching
    result = [word for word in jobtitle.split() if (word not in negative_list) or (word in positive_list)]
    result2str = ' '.join(result)

    return result2str

class JobTitleStringClean(object):
    """
    Clean job titles
    """

    def __init__(self):
        self.dict = negative_positive_dict()
        self.negative_list = self.dict['places'] + self.dict['states']
        self.positive_list = self.dict['onetjobs']

    def clean(self, df_jobtitles, groupby_keys):
        """
        Clean the job titles by rules and negative dictionary.
        Args:
            df_jobtitles: job titles in pandas DataFrame
            groupby_keys: a list of keys to be grouped by
        Returns:
            agg_cleaned_jobtitles: a clenaed and aggregated verison of job title in pandas DataFrame

        """
        df_jobtitles = df_jobtitles.fillna('without jobtitle')

        columns = list(df_jobtitles.columns)
        cleaned_jobtitles = OrderedDict({key: [] for key in columns})
        for i, row in enumerate(df_jobtitles.values):
            try:
                for colname in columns:
                    if colname == 'title':
                        new_title = clean_by_rules(row[columns.index(colname)])
                        new_title = clean_by_neg_dic(new_title, self.negative_list, self.positive_list)
                        cleaned_jobtitles[colname].append(new_title)
                    else:
                        cleaned_jobtitles[colname].append(row[columns.index(colname)])
            except TypeError:
                print('There is some TypeError',row)

        cleaned_jobtitles = pd.DataFrame(cleaned_jobtitles)
        agg_cleaned_jobtitles = pd.DataFrame(cleaned_jobtitles.groupby(groupby_keys, as_index=False)['count'].sum())
        agg_cleaned_jobtitles = agg_cleaned_jobtitles.fillna('without jobtitle')

        return agg_cleaned_jobtitles



