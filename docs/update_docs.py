import os
import pkgutil
import importlib
import yaml

import skills_ml.algorithms
import skills_ml.datasets
import skills_ml.evaluation
import skills_ml.job_postings

from nbconvert.nbconvertapp import NbConvertApp


def generate_api_docs():
    pkgs = [
        (skills_ml.algorithms, 'Algorithms'),
        (skills_ml.datasets, 'External Dataset Processors'),
        (skills_ml.evaluation, 'Evaluation Tools'),
        (skills_ml.job_postings, 'Job Posting Dataset Processors'),
    ]
    generate = []
    pages = []
    doc_page = {}
    for pkg_module, pkg_title in pkgs:
        pkgpath = os.path.dirname(pkg_module.__file__)
        pkg_modules = [
            importlib.import_module(pkg_module.__name__ + '.' + module_name)
            for _, module_name, _ in pkgutil.iter_modules([pkgpath])
        ]
        doc_page[pkg_title] = []
        generated_page = {}
        fname = pkg_module.__name__ + '.md'
        generated_page[fname] = []
        doc_page[pkg_title] = fname

        for module in pkg_modules:
            generated_page[fname].append(module.__name__ + '+')
            modulepath = os.path.dirname(module.__file__)
            sub_modules = []
            for _, module_name, _ in pkgutil.iter_modules([modulepath]):
                try:
                    print(module.__name__ + '.' + module_name)
                    sm = importlib.import_module(module.__name__ + '.' + module_name)
                    sub_modules.append(sm)
                except ImportError:
                    continue
            for sub_module in sub_modules:
                generated_page[fname].append(sub_module.__name__ + '+')

        generate.append(generated_page)
    with open('pydocmd.yml', 'r+') as f:
        pydocyml_config = yaml.load(f)
        pydocyml_config['generate'] = generate
        pages = pydocyml_config['pages']
        for doc_page_title, doc_pages in doc_page.items():
            found = False
            for page in pages:
                if doc_page_title in page:
                    page[doc_page_title] = doc_pages
                    found = True
            if not found:
                new_thing = {}
                new_thing[doc_page_title] = doc_pages
                pages.append(new_thing)
        f.seek(0)
        f.write(yaml.dump(pydocyml_config))
        f.truncate()


def convert_tour_nb_to_document():
    app = NbConvertApp()
    app.initialize()
    app.notebooks = ['../Skills-ML Tour.ipynb']
    app.export_format = 'markdown'
    app.output_base = 'skills_ml_tour.md'
    app.writer.build_directory = 'sources/'
    app.convert_notebooks()


if __name__ == "__main__":
    convert_tour_nb_to_document()
    generate_api_docs()
