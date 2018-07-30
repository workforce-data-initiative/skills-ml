from .base import JobGeographyQuerier


class JobStateQuerier(JobGeographyQuerier):
    @property
    def name(self):
        return 'state'

    @property
    def output_columns(self):
        return (
            ('state', 'US state of the job posting as given in the addressRegion field'),
        )

    def _query(self, job_posting):
        return (job_posting.get('jobLocation', {}).get('address', {}).get('addressRegion', None),)
