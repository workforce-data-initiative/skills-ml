import io
import logging
import os
import requests
import zipfile


class OnetSourceDownloader(object):
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
