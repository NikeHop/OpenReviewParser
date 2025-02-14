# OpenReview Parser

![Mypy](https://img.shields.io/badge/mypy-checked-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![pydocstyle](https://img.shields.io/badge/pydocstyle-passing-brightgreen)

This repo contains a pipeline that parses the submissions and reviews of all OpenReview venues into a unified format and annotates them with metadata including:

* research_hypothesis (annotated via LLM)
* references (from [Semantic Scholar](https://www.semanticscholar.org/))
* citation counts for accepted papers (from [Semantic Scholar](https://www.semanticscholar.org/))


## Dependencies 

Create a new conda environment and install the dependencies 

```
conda create -n openreview_parser python=3.11
conda activate openreview_parser
pip install -e . 
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

An overview of the unifying data model for submissions and reviews can be found [here](./openreview_parser/utils/README.md).

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
Run the pipeline from the `./openreview_parser/pipeline` directory:

```
bash ./scripts/run_pipeline.sh
```

**Note 1:** It will run the pipeline twice. Once to access venues only accessible via the OpenReview API V1 and once for the venues accessible via the OpenReview API V2. 

**Note 2:** The metadata annotation steps are optional and can be toggled on/off via in the `./configs/pipeline.yaml`. The default is no metadata annotation

### Metadata annotation

* **References**: First from the `./openreview_parser/scientific_databases` directory run

```
python s2_datasets.py
```

This requires around 140G of disk space and a Semantic Scholar API key. The paper and abstract datasets of Semantic Scholar are downloaded to build a mapping from titles to paper info for the retrieval of reference informations. 

* **Hypothesis Annotation**: Requires to set OPENAI_API_KEY as environment variable with the corresponding key.

* **Citation Count**: Semantic Scholar key recommended otherwise rate limits are easily reached. 

 
### Run pipeline for individual venues 

For venues accessible via API V1:

```
python pipeline.py --config ./configs/pipeline_v1.yaml --venue "ICLR.cc/2022/Conference"
```

For venues accessible via API V2:

```
python pipeline.py --config ./configs/pipeline_v2.yaml --venue "ICLR.cc/2024/Conference"
```



## Dataset 

A dataset derived from running the pipeline can be found on HF: [dataset](https://huggingface.co/datasets/nhop/scientific-quality-score-prediction).

## Updates 

Timestamps at which the pipeline was run (datasets partially available [here](nhop/scientific-quality-score-prediction))

* Latest: 1.1.2025
