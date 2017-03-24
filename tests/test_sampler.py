from algorithms.file_sampler import sampler
from unittest.mock import mock_open
import os
import tempfile

test_csv = """
test manager audits and governance,3
immediate job opportunities available,1
aquisition specialist recruiter,1
collections operations specialist,1
maintenance manager chief engineer,1
manager trainee store opening soon,1
sports minded grad entry level sales marketing,4
catering coordinator opening soon port,1
field admissions representative,1
executive training program entry level,1
marketing copywriter,4
medical collections appeals and denials,2
retail store management and sales associates store to open,1
macys pump town retail commission sa,1
sales engineering manager design prototyping,1
relationship manager metro,1
reo property management coordinator,2
landman,5
small engine repair technicians mechanic,2
"""

SAMPLENUM = 5

def test_job_title_sampler():
    handle, filename = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(handle, 'w', encoding='utf-8') as f:
        f.write(test_csv)

    assert sampler.reservoir_sample(SAMPLENUM, filename, 42) != sampler.reservoir_sample(SAMPLENUM, filename, 24)
