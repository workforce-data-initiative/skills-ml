import csv
import pandas as pd
import logging

from utils.nlp import NLPTransforms


class OnetSkillExtractor(object):
    """
    An object that creates a skills CSV based on ONET data
    """
    def __init__(self, onet_source, output_filename, hash_function):
        """
        Args:
            output_filename: A filename to write the final dataset
            onet_source: An object that is able to fetch ONET files by name
            hash_function: A function that can hash a given string
        """
        self.output_filename = output_filename
        self.onet_source = onet_source
        self.hash_function = hash_function

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
        nlp = NLPTransforms()
        # create dataframes for each KSA type
        standard_columns = ['O*NET-SOC Code', 'Element ID', 'Element Name']
        skills = self.onet_to_pandas('Skills.txt', standard_columns, 'skill')
        ability = self.onet_to_pandas('Abilities.txt', standard_columns, 'ability')
        knowledge = self.onet_to_pandas('Knowledge.txt', standard_columns, 'knowledge')
        tools = self.onet_to_pandas(
            'Tools and Technology.txt',
            ['O*NET-SOC Code', 'Commodity Code', 'T2 Example'],
            'tool',
            use_relevance=False
        )

        # Concat KSA dataframes into one table
        # note significant duplications since it's by ONET SOC Code
        new_columns = ['O*NET-SOC Code', 'Element ID', 'ONET KSA', 'ksa_type']
        skills.columns = new_columns
        ability.columns = new_columns
        knowledge.columns = new_columns
        tools.columns = new_columns

        # ... concat KSA descriptions
        # (useful for users wanting to know what this skill is about)
        onet_modelreference = self.onet_to_pandas(
            'Content Model Reference.txt',
            ['Element ID', 'Description'],
            ksa_type=None,
            use_relevance=False
        )

        onet_ksas = pd.concat(
            (skills, ability, knowledge, tools),
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
        onet_ksas[nlp.transforms[0]] = onet_ksas['ONET KSA']\
            .apply(nlp.lowercase_strip_punc)

        onet_ksas.to_csv(self.output_filename, sep='\t')
