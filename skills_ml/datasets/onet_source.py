"""Download ONET files from their site"""
import requests
import os
import logging
import io
import zipfile


class OnetToMemoryDownloader(object):
    """Downloads newest version of ONET as of time of writing and returns it as text"""
    url_prefix = 'http://www.onetcenter.org/dl_files/database/db_22_3_text'

    def download(self, source_file):
        url = f'{self.url_prefix}/{source_file}.txt'
        response = requests.get(url)
        return response.text


class OnetToDiskDownloader(object):
    url_prefix = 'http://www.onetcenter.org/dl_files'
    directory = 'tmp/'

    def download(self, version, source_file, output_filename):
        destination_filename = self.directory + version + '_' + output_filename
        if not os.path.isfile(destination_filename):
            logging.info('Could not find %s, downloading', destination_filename)
            url_suffix = '{}.zip'.format(version)
            if version > 'db_20_0':
                zip_file_url = '/'.join([self.url_prefix, 'database', url_suffix])
            else:
                zip_file_url = '/'.join([self.url_prefix, url_suffix])
            logging.info(zip_file_url)
            response = requests.get(zip_file_url)
            z = zipfile.ZipFile(io.BytesIO(response.content))
            source_filename = '{}/{}'.format(version, source_file)
            logging.info('Extracting occupation data')
            with z.open(source_filename) as input_file:
                with open(destination_filename, 'wb') as output_file:
                    output_file.write(input_file.read())
        return destination_filename


onet_major_group = {
            '11': 'Management Occupations',
            '13': 'Business and Financial Operations Occupations',
            '15': 'Computer and Mathematical Occupations',
            '17': 'Architecture and Engineering Occupations',
            '19': 'Life, Physical, and Social Science Occupations',
            '21': 'Community and Social Service Occupations',
            '23': 'Legal Occupations',
            '25': 'Education, Training, and Library Occupations',
            '27': 'Arts, Design, Entertainment, Sports, and Media Occupations',
            '29': 'Healthcare Practitioners and Technical Occupations',
            '31': 'Healthcare Support Occupations',
            '33': 'Protective Service Occupations',
            '35': 'Food Preparation and Serving Related Occupations',
            '37': 'Building and Grounds Cleaning and Maintenance',
            '39': 'Personal Care and Service Occupations',
            '41': 'Sales and Related Occupations',
            '43': 'Office and Administrative Support Occupations',
            '45': 'Farming, Fishing, and Forestry Occupations',
            '47': 'Construction and Extraction Occupations',
            '49': 'Installation, Maintenance, and Repair Occupations',
            '51': 'Production Occupations',
            '53': 'Transportation and Material Moving Occupations',
            '55': 'Military Specific Occupations'
        }
