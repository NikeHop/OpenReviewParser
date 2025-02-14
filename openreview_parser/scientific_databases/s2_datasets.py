"""
Merge the abstract and papers info dataset from Semantic Scholar to create title2s2_paper_info dataset.

The following steps are performed:
1. Download the abstract and papers info dataset from Semantic Scholar.
2. Filter the papers to include only Computer Science papers.
3. Add the abstracts to the papers.
4. Chunk the title2s2_paper_info dataset.
"""

import glob
import gzip
import json
import os

import requests
import tqdm
import wget

from openreview_parser.scientific_databases.s2 import get_s2_header


def download_datasets():
    """
    Download the abstract and paper info datasets from the Semantic Scholar API.

    The datasets are downloaded and saved in the 'data/s2' directory. The 'abstracts' and 'paper_info' subdirectories
    are created if they don't exist. The datasets are downloaded using the Semantic Scholar API and saved as compressed
    files in the respective subdirectories.

    Note: This function requires an API key to access the Semantic Scholar API.

    Returns:
        None
    """
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
    """
    Filter and extracts information from S2 datasets for Computer Science papers.

    This function reads S2 dataset files, filters out papers that are related to Computer Science,
    and extracts relevant information such as title, authors, and corpus ID. The extracted information
    is stored in a dictionary called `corpus_id2title` and saved to a JSON file.

    Note: This function assumes the presence of S2 dataset files in the specified data directory.

    Returns:
        None
    """
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
    """
    Add abstracts to the title2paper_info dictionary.

    This function loads the title2paper_info and corpus_id2title dictionaries from JSON files.
    It then iterates over a collection of abstract files, extracts relevant information, and adds it to the title2paper_info dictionary.
    Finally, it saves the updated title2paper_info dictionary back to a JSON file.

    Note: The function assumes the presence of specific file paths and file structures.

    Returns:
        None
    """
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
                print(elem)
                corpus_id = str(elem["corpusid"])
                if corpus_id not in corpus_id2title:
                    continue
                info = corpus_id2title[corpus_id]
                title = info["title"]
                title2paper_info[title] = {}
                title2paper_info[title]["title"] = info["title"]
                title2paper_info[title]["corpusid"] = info["corpusid"]
                title2paper_info[title]["authors"] = info["authors"]
                title2paper_info[title]["abstract"] = elem["abstract"]
                title2paper_info[title]["external_ids"] = elem["openaccessinfo"][
                    "externalids"
                ]

        with open("data/s2/title2paper_info.json", "w") as file:
            json.dump(title2paper_info, file, indent=4)


def chunk():
    """
    Chunk the title2paper_info dictionary into smaller chunks and save them as separate JSON files.

    This function creates a directory 'data/s2/parsed_paper_info' if it doesn't exist. It then loads the 'title2paper_info'
    dictionary from the file 'data/s2/title2paper_info.json'. The dictionary is transformed to have lowercase stripped titles
    as keys. The list of all titles is saved as 'data/s2/all_titles.json'.

    The 'title2paper_info' dictionary is chunked into smaller dictionaries, each containing 'chunk_size' number of titles.
    Each chunk is saved as a separate JSON file in the 'data/s2/parsed_paper_info' directory. The 'key_dictionary' is also
    created, which maps the chunk number to the first and last title in that chunk. The 'key_dictionary' is saved as
    'data/s2/key_dictionary.json'.

    Returns:
        None
    """
    os.makedirs("data/s2/parsed_paper_info", exist_ok=True)

    # Load title2paperinfo
    with open("data/s2/title2paper_info.json", "r") as file:
        title2paper_info = json.load(file)

    title2paper_info = {
        title.strip().lower(): title2paper_info[title] for title in title2paper_info
    }

    all_titles = list(title2paper_info.keys())

    with open("data/s2/all_titles.json", "w") as file:
        json.dump(all_titles, file, indent=4)

    all_titles = list(sorted(all_titles))

    # Chunk the title2paper_info
    chunk_size = 5000
    n_chunks = 0
    current_position = 0
    key_dictionary = {}
    pbar = tqdm.tqdm(total=len(all_titles))
    while current_position < len(all_titles):
        titles = all_titles[current_position : current_position + chunk_size]
        first_title = titles[0]
        last_title = titles[-1]
        key_dictionary[n_chunks] = [first_title, last_title]
        paper_info_data = {title: title2paper_info[title] for title in titles}

        with open(
            os.path.join("./data/s2/parsed_paper_info", f"{n_chunks}.json"), "w"
        ) as file:
            json.dump(paper_info_data, file, indent=4)

        n_chunks += 1
        current_position += chunk_size
        pbar.update(chunk_size)

    with open("data/s2/key_dictionary.json", "w") as file:
        json.dump(key_dictionary, file, indent=4)


def create_title2s2_paper_info():
    """Download datasets, filter papers, add abstracts, and chunk the data."""
    download_datasets()
    filter_papers()
    add_abstracts()
    chunk()


if __name__ == "__main__":
    assert (
        os.environ.get("SEMANTIC_SCHOLAR_API_KEY", None) is not None
    ), "Semantic Scholar API key not found"
    create_title2s2_paper_info()
