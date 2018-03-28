# Usage Examples

## [Corpus Creator with Sampling and Filtering](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/Creating Corpus and Sampled Corpus.ipynb)

To showcase the corpus creator and its options, we generate a few different job postings corpora:

- a simple one from a single quarter's worth of data
- filtered on different fields like SOC code and base salary
- reservoir-sampled versions of each of the above

## [Extracting Skills using Noun Phrase Endings](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/NounPhraseSkillExtraction.py)

To showcase the noun phrase skill extractor, we download open job postings
from Virginia Tech's open data portal and run them through the skill extractor.
In the end, we have the most commonly occurring noun phrases ending in
'skill' or 'skills'.

## [Generate Skill Candidates for Further Evaluation](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/UploadCandidatesFromSample.py)

To showcase how skill extraction algorithms can be tested, we run extraction several times with different parameters:

- Skill extraction algorithms (exact, fuzzy matching)
- Base skill lists (ONET abilities, ONET skills, ONET knowledge)
- Samples (a 300 job posting sample, a 10k job posting sample)

For each combination of the above parameters, we upload the extracted skill candidates to S3 for further evaluation, for instance by a human labeller. In addition, this example shows how to parallelize the skill extraction.

## [Train an Word2Vec Embedding Model using Quarterly Jobposting Data](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/TrainEmbedding.py)

To showcase the interface of training a word2vec embedding model in an online batch learning fashion:

- A list of quarters for creating the corpus from job posting data
- A trainer object that specifies some parameters of source, s3 path, batch size, model type ...etc.
- The train method takes whatever arugments `gensim.models.word2vec.Word2Vec` or `gensim.model.doc2vec.Doc2Vec` has

## [Compute and Aggregate Properties of Job Postings as a Tabular Dataset](https://github.com/workforce-data-initiative/skills-ml/blob/master/examples/ComputeAndAggregateJobPostingProperties.py)

To show job posting property computation and aggregation,
we calculate job posting counts by cleaned title, and upload
the resulting CSV to S3.

This is essentially a mini version of the Data@Work Research Hub.

To enable this example to be run with as few dependencies as possible, we use:

- a fake local s3 instance
- a sample of the Virginia Tech open job postings dataset
- only title cleaning and job counting.
