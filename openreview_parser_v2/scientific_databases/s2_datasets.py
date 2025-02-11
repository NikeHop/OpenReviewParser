"""
Merge the abstract and papers info dataset from Semantic Scholar
to create title2s2_paper_info dataset.
Only inlude Computer Science papers.
"""

import glob
import gzip
import json
import os

import requests
import tqdm
import wget

from openreview_parser_v2.scientific_databases.s2 import get_s2_header


def download_datasets():
    # Create directory
    data_directory = "data/s2"
    os.makedirs(os.path.join(data_directory, "abstracts"), exist_ok=True)
    os.makedirs(os.path.join(data_directory, "paper_info"), exist_ok=True)

    header = get_s2_header(use_api_key=True)

    # Download the abstract dataset
    response = requests.get(
        "https://api.semanticscholar.org/datasets/v1/release/latest/dataset/abstracts",
        headers=header,
    ).json()

    for i, file in enumerate(response["files"]):
        goal_filepath = os.path.join(
            os.path.join(data_directory, "abstracts"), f"{i}.gz"
        )
        wget.download(file, out=goal_filepath)

    # Download the papers info dataset
    response = requests.get(
        "https://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers",
        headers=header,
    ).json()

    for i, file in enumerate(response["files"]):
        goal_filepath = os.path.join(
            os.path.join(data_directory, "paper_info"), f"{i}.gz"
        )
        wget.download(file, out=goal_filepath)


def filter_papers():
    # Create title2corpus_id for Computer Science papers.
    if os.path.exists("data/s2/corpus_id2title.json"):
        with open("data/s2/corpus_id2title.json", "r") as file:
            corpus_id2title = json.load(file)
    else:
        corpus_id2title = {}

    data_directory = "data/s2/paper_info"
    files = glob.glob(data_directory + "/*.gz")
    for filepath in tqdm.tqdm(files):
        with gzip.open(filepath, "rt") as file:
            for line in file:
                elem = json.loads(line)
                if elem["s2fieldsofstudy"] != None:
                    fields_of_study = [
                        elem["category"] for elem in elem["s2fieldsofstudy"]
                    ]
                    if "Computer Science" in fields_of_study:
                        corpus_id2title[elem["corpusid"]] = {
                            "title": elem["title"].lower(),
                            "authors": [author["name"] for author in elem["authors"]],
                            "corpusid": elem["corpusid"],
                        }

        # Save the corpus_id2title
        with open("data/s2/corpus_id2title.json", "w+") as file:
            json.dump(corpus_id2title, file, indent=4)


def add_abstracts():
    # Load title2paperinfo
    if os.path.exists("data/s2/title2paper_info.json"):
        with open("data/s2/title2paper_info.json", "r") as file:
            title2paper_info = json.load(file)
    else:
        title2paper_info = {}

    # Load title2corpus_id
    with open("data/s2/corpus_id2title.json", "r") as file:
        corpus_id2title = json.load(file)

    data_directory = "data/s2/abstracts"
    files = glob.glob(data_directory + "/*.gz")
    for filepath in files:
        with gzip.open(filepath, "rt") as file:
            data = [json.loads(line) for line in file]
            for elem in tqdm.tqdm(data):
                if elem["corpusid"] not in corpus_id2title:
                    continue
                info = corpus_id2title[elem["corpusid"]]
                title = info["title"]
                title2paper_info[title] = {}
                title2paper_info[title]["title"] = info["title"]
                title2paper_info[title]["corpusid"] = info["corpusid"]
                title2paper_info[title]["authors"] = info["authors"]
                title2paper_info[title]["abstract"] = elem["abstract"]
                title2paper_info[title]["external_ids"] = elem["external_ids"]

        with open("data/s2/title2paper_info.json", "w") as file:
            json.dump(title2paper_info, file, indent=4)


def create_title2s2_paper_info():
    add_abstracts()


if __name__ == "__main__":
    assert (
        os.environ.get("SEMANTIC_SCHOLAR_API_KEY", None) is not None
    ), "Semantic Scholar API key not found"
    create_title2s2_paper_info()
