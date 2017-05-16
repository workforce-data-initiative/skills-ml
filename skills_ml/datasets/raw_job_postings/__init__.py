from skills_ml.datasets.raw_job_postings.virginia import VirginiaTransformer
from skills_ml.datasets.raw_job_postings.usajobs import USAJobsTransformer

importers = {'VA': VirginiaTransformer, 'US': USAJobsTransformer}
