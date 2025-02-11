# OpenReview Parser V2

![Mypy](https://img.shields.io/badge/mypy-checked-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains a pipeline that parses the submissions and reviews of all OpenReview venues accessible with the API V2 into a unified format and annotates them with metadata including:

* research_hypothesis (annotated via LLM)
* references (from [Semantic Scholar](https://www.semanticscholar.org/))
* citation counts for accepted papers (from [Semantic Scholar](https://www.semanticscholar.org/))


**Note:** For venues requiring API V1 see [here]().

## Dependencies 

Create a new conda environment and install the dependencies 

```
conda create -n openreview_parser2 
```

#### GROBID 

We parse the pdfs of submissions using [GROBID](https://github.com/kermitt2/grobid). To setup GROBID run from the `./openreview_parser/pipeline` directory:

```
bash ./scripts/setup_grobid.sh
```
To test whether the setup worked run:

```
bash ./scripts/run_grobid.sh
```

If you run into trouble setting up GROBID some git issues from the followin [repo](https://github.com/allenai/s2orc-doc2json) might be helpful.

## Data Model

An overview of the unifying data model for submissions and reviews can be found [here](./openreview_parser_v2/utils/README.md).

## Run pipeline 


The pipeline proceeds in the following steps:

* Retrieve all venues from OpenReview
* For each venue where submission and reviews are available build a schema of the venue and map the schema to the unified data model
* Get all submissions and reviews and parse them into the unified data model 
* Annotate the submissions with metadata:
    * Title and Abstract of References
    * Research Hypothesis 
    * Citation Counts

#### Run Script
Run the pipeline from the `./openreview_parser_v2/pipeline` directory:

```
bash ./scripts/run_pipeline.sh
```

**Note 1:** The metadata annotation steps are optional and can be toggled on/off via in the `./configs/pipeline.yaml`

**Note 2:** For steps that require querying Semantic Scholar an API key is recommended. Rate limits will be easily reached leading leading to metadata being None.
 


## Dataset 

A dataset derived from running the pipeline can be found on HF: [dataset](https://huggingface.co/datasets/nhop/scientific-quality-score-prediction).

## Updates 

Timestamps at which the pipeline was run (datasets partially available [here](nhop/scientific-quality-score-prediction))

* Latest: 1.1.2025
