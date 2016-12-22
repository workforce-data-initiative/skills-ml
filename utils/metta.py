from datetime import date
from calendar import monthrange
import metta
import pandas as pd
from random import randint

def quarter_boundaries(quarter):
    year, quarter = quarter.split('Q')
    year = int(year)
    quarter = int(quarter)
    first_month_of_quarter = 3 * quarter - 2
    last_month_of_quarter = 3 * quarter
    first_day = date(year, first_month_of_quarter, 1)
    last_day = date(year, last_month_of_quarter, monthrange(year, last_month_of_quarter)[1])
    return first_day, last_day


def metta_config(quarter, num_dimensions):
    first_day, last_day = quarter_boundaries(quarter)
    return {
        'start_time': first_day,
        'end_time': last_day,
        'prediction_window': 3, # ???
        'label_name': 'onet_soc_code',
        'label_type': 'categorical',
        'matrix_id': 'job_postings_{}'.format(quarter),
        'feature_names': ['doc2vec_{}'.format(i) for i in range(num_dimensions)],
    }

def upload_to_metta(train_path, test_path, train_quarter, test_quarter, num_dimensions):
    train_config = metta_config(train_quarter, num_dimensions)
    test_config = metta_config(test_quarter, num_dimensions)

    X_train = pd.read_csv(train_path,sep=',')
    X_train['label'] = pd.Series([randint(0,23) for i in range(len(X_train))])
    X_test = pd.read_csv(test_path, sep=',')
    X_test['label'] = pd.Series([randint(0,23) for i in range(len(X_test))])
    #Y_train = pd.read_csv()
    metta.archive_train_test(
        train_config,
        X_train,
        test_config,
        X_test,
        directory='wdi'
    )

if __name__ == '__main__':
    upload_to_metta('../tmp/job_features_train_2011Q1.csv',
                    '../tmp/job_features_test_2016Q1.csv',
                    '2011Q1',
                    '2016Q1',
                    500)



