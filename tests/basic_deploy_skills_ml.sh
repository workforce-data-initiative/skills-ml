# Set up Airflow Directory variable if not already defined
export AIRFLOW_HOME=${AIRFLOW_HOME-$HOME/airflow/}

# Install Airflow with minimal requirements (access to S3 bucket)
pip install airflow[s3]

# Airflow examples required Hive operator to be installed
# but we don't neccesarily have it, so we do not load examples
sed -i '/load_examples = True/c\load_examples = False' ~/airflow/airflow.cfg

# Point airflow to the WDI Dags (this is a hack but whatever)
read -p "Please enter full path to local skills-ml/ respository (e.g. /home/foo/WDI/): " local_repo
sed -i '/dags_folder = */c\dags_folder = '"$local_repo"'skills-ml/dags' ~/airflow/airflow.cfg

# Airflow needs a path from which to look for skills-ml library
export PYTHONPATH=$PYTHONPATH:$local_repo

# Initalize Airflow database on first run, kick off a scheduler (sequential) and webserver
# Also shutdown any current webserver already running
cat $AIRFLOW_HOME/airflow-webserver.pid | xargs kill -9
airflow initdb
airflow scheduler &
airflow webserver -hn 127.0.0.1 -p 8080 &
google-chrome-stable --args 'http://127.0.0.1:8080'
echo "Set up your S3 Credentials (with name s3_default, or leave blank) according to: http://stackoverflow.com/questions/39997714/airflow-s3-connection-using-ui"
