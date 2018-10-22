"""Process ONET skill lists of various types into a common format"""
import csv
import pandas as pd
import logging

from skills_ml.algorithms.nlp import transforms, lowercase_strip_punc


KSA_TYPE_CONFIG = {
    'skill': ('Skills.txt', ['O*NET-SOC Code', 'Element ID', 'Element Name'], 'skill'),
    'ability': ('Abilities.txt', ['O*NET-SOC Code', 'Element ID', 'Element Name'], 'ability'),
    'knowledge': ('Knowledge.txt', ['O*NET-SOC Code', 'Element ID', 'Element Name'], 'knowledge'),
    'tool': ('Tools and Technology.txt', ['O*NET-SOC Code', 'Commodity Code', 'T2 Example'], 'tool', False)
}

class OnetSkillListProcessor(object):
    """
    An object that creates a skills CSV based on ONET data

    Originally written by Kwame Porter Robinson
    """
    def __init__(self, onet_source, output_filename, hash_function, ksa_types=None):
        """
        Args:
            output_filename: A filename to write the final dataset
            onet_source: An object that is able to fetch ONET files by name
            hash_function: A function that can hash a given string
            ksa_types: A list of onet skill types to include.
                All strings must be keys in KSA_TYPE_CONFIG.
                Defaults to all keys in KSA_TYPE_CONFIG
        """
        self.output_filename = output_filename
        self.onet_source = onet_source
        self.hash_function = hash_function
        self.ksa_types = ksa_types or KSA_TYPE_CONFIG.keys()

    def onet_to_pandas(self, filename, col_name, ksa_type, use_relevance=True):
        """
        Args:
            filename: an unpathed filename referring to an ONET skill file
            col_name: A list of columns to extract from the file
            use_relevance (optional): Whether or not to filter out irrelevant
                rows. Defaults to True.

        Returns:
            A pandas DataFrame
        """
        logging.info('Converting ONET %s to pandas', filename)
        with self.onet_source.ensure_file(filename) as fullpath:
            with open(fullpath) as f:
                if use_relevance:
                    # we want those rows that
                    # - use LV measure
                    # - are relevant and
                    # - have a rating at least greater than 0
                    onet = [
                        row for row in csv.DictReader(f, delimiter='\t')
                        if row['Scale ID'] == 'LV'
                        and row['Not Relevant'] == 'N'
                        and float(row['Data Value']) > 0
                    ]
                else:
                    onet = [row for row in csv.DictReader(f, delimiter='\t')]
        onet = pd.DataFrame(onet)
        if ksa_type:
            col_name = col_name + ['ksa_type']
            onet['ksa_type'] = ksa_type

        for col in col_name:
            onet[col] = onet[col].astype(str).str.lower()
        return onet[col_name]


    def run(self):
        """
            Creates a skills CSV based on ONET KSAT data,
            with descriptions fetched from the content model reference
        """
        # create dataframes for each KSA type
        dataframes = [
            self.onet_to_pandas(*(KSA_TYPE_CONFIG[ksa_type]))
            for ksa_type in self.ksa_types
        ]

        # Concat KSA dataframes into one table
        # note significant duplications since it's by ONET SOC Code
        new_columns = ['O*NET-SOC Code', 'Element ID', 'ONET KSA', 'ksa_type']
        for df in dataframes:
            df.columns = new_columns

        # ... concat KSA descriptions
        # (useful for users wanting to know what this skill is about)
        onet_modelreference = self.onet_to_pandas(
            'Content Model Reference.txt',
            ['Element ID', 'Description'],
            ksa_type=None,
            use_relevance=False
        )

        onet_ksas = pd.concat(
            dataframes,
            ignore_index=True
        )
        onet_ksas = pd.merge(
            onet_ksas,
            onet_modelreference,
            how='left',
            on=['Element ID']
        )

        logging.info('Uniqifying skills')
        onet_ksas.drop_duplicates('ONET KSA', inplace=True)
        onet_ksas['skill_uuid'] = onet_ksas['ONET KSA']\
            .apply(self.hash_function)
        onet_ksas[transforms[0]] = onet_ksas['ONET KSA']\
            .apply(lowercase_strip_punc)

        onet_ksas.to_csv(self.output_filename, sep='\t')
