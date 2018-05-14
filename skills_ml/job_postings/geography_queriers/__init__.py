"""Extracting geographies from job posting datasets"""
import us

STATE_NAME_LOOKUP = us.states.mapping('abbr', 'name')


def job_posting_search_strings(job_posting):
    """Convert a job posting to a geocode-ready search string

    Includes city and state if present, or just city

    Args:
        job_posting (dict) A job posting in schema.org/JobPosting json form

    Returns: (string) A geocode-ready search string
    """
    location = job_posting.get('jobLocation', None)
    if not location:
        return []
    locality = location.get('address', {}).get('addressLocality', None)
    region = location.get('address', {}).get('addressRegion', None)
    if locality and region:
        # lookup state name, if it's not there just use whatever they typed

        lookups = ['{}, {}'.format(locality, region)]
        formatted_region = STATE_NAME_LOOKUP.get(region, None)
        if formatted_region:
            lookups.append('{}, {}'.format(locality, formatted_region))
        return lookups
    elif locality:
        return [locality]
    else:
        return []
