"""
Obtaining the submissions from the OpenReview data model and parse them into the data model.
"""

import glob
import logging 
import os 
import time 

import openreview 
import tqdm 

from urllib.error import HTTPError
from urllib.request import urlretrieve

from openreview_parser.scientific_databases.openreview_v2 import get_submissions
from openreview_parser.utils.data import VenueInstance



def parse_submissions(venue_instances:list[VenueInstance], openreview_client:openreview.Client, config:dict) -> None:
    existing_venue_dataset = load_existing_venue_datasets()

    for venue_instance in venue_instances:
        venue_id = venue_instance.venue.replace("/", "_").replace(".", "_")
        
        if venue_id in existing_venue_dataset:
            logging.info(f"Skipping {venue_instance.venue} as it already exists")
            continue
    
        logging.info(f"Processing {venue_instance.venue}")
        submissions = get_submissions(openreview_client, venue_instance, config)

        if len(submissions) == 0:
            logging.info(f"No submissions found for {venue_instance.venue}")
            continue
        
        parse_submission_pdfs(
            client,
            dataset,
            note_type_mapping,
            submissions,
            venue,
            statistics,
            config,
        )


def load_existing_venue_datasets():
    directory = "./"
    venue_datasets = glob.glob(os.path.join(directory, "venue_datasets", "*.json"))
    existing_venue_datasets = []

    # For each filename extract the venue id
    for filename in venue_datasets:
        venue_id = filename.split("/")[-1].split(".")[0]
        existing_venue_datasets.append(venue_id)
    
    return existing_venue_datasets

def parse_submissions(submissions: list[openreview.Note],venue_id:str,config:dict) -> tuple[dict, int]:
    """
    Parses a submission and extracts relevant information such as PDF, reviews, comments, and decisions.

    Args:
        client (openreview.Client): The OpenReview client object.
        note_type_mapping (dict): A mapping of note types to their corresponding labels.
        sub (dict): The submission to parse.
        venue (str): The venue of the submission.
        config (dict): Configuration settings.

    Returns:
        dataset (dict): The dataset containing the submission information.
        n_reviews (int): The number of reviews for the submission.
    """

    for submission in tqdm.tqdm(submissions):
        id = submission["id"]
        forum_id = submission["forum"]

        if "paperhash" not in submission["content"]:
            logging.info(f"Submission {id} in venue {venue_id} does not have a paperhash.")
            return None

        paperhash = submission["content"]["paperhash"]["value"]

        # Get pdf
        pdf_success = False
        while not pdf_success:
            try:
                filename = os.path.join(config["save_directory"], "pdfs", f"{paperhash}.pdf")
                if not os.path.exists(filename):
                    url = f"https://openreview.net/pdf?id={id}"
                    urlretrieve(url, filename)
                pdf_success = True
            except HTTPError as e:
                if e.code == 429:
                    logging.info("Too many requests. Sleep for 5 seconds")
                    time.sleep(5)
                else:
                    logging.warning(f"Could not download the pdf for submission {id}.")
                    logging.warning(e)
                    return {}, 0

            except Exception as e:
                logging.warning(f"Could not download the pdf for submission {id}.")
                logging.warning(e)
                return {}, 0

    # Get reviews and comments and meta_review
    notes = get_notes_with_backoff_post_2024(client, forum=forum_id)

    review_ids, comment_ids = [], []

    for note in notes:
        note_type = note.invitations[0].split("/")[-1]

        if note_type == "Blind_Submission":
            continue

        if note_type not in note_type_mapping:
            logging.warning(
                f"Trying to parse an unknown Note Type: {note_type} for submission: {id} at venue {venue}"
            )
            continue

        note_type = note_type_mapping[note_type]
        if note_type == "Review":
            review_id = note.id
            dataset["reviews"][review_id] = note.to_json()
            review_ids.append(review_id)
            n_reviews += 1

        if note_type == "Comment":
            comment_id = note.id
            dataset["comments"][comment_id] = note.to_json()
            comment_ids.append(comment_id)

        if note_type == "Decision":
            decision_id = note.id
            dataset["decisions"][decision_id] = note.to_json()
            sub["decision_id"] = decision_id

    sub["comment_ids"] = comment_ids
    sub["review_ids"] = review_ids
    dataset["submissions"][id] = sub

    return dataset, n_reviews


def parse_submission_pdfs(dataset: dict, config: dict) -> None:
    """
    Parse submission PDFs and update the dataset with parsed PDF information.

    Args:
        dataset (dict): The dataset containing submissions.
        config (dict): The configuration settings.

    Returns:
        None
    """
    pdf_directory = os.path.join(config["save_directory"], "pdfs")

    start = time.time()
    if config["aws"]["enabled"]:
        # Initialize S3 client
        files = [
            sub["content"]["paperhash"]["value"] + ".pdf"
            for sub in dataset["submissions"].values()
            if os.path.exists(
                os.path.join(pdf_directory, sub["content"]["paperhash"]["value"] + ".pdf")
            )
        ]
        input_directory = config["save_directory"] + "/pdfs"
        output_directory = "pdfs"
        if config["batch"]:
            upload_files_s3_batch(
                files, input_directory, output_directory, config
            )
        else:
            upload_files_s3(files, input_directory, output_directory, config)
    end = time.time()
    logging.info(f"Loading pdfs to S3 took {end - start} seconds")

    start = time.time()
    if config["batch"]:
        paperhashes = []
        pdf_files = []
        for submission_id, submission in dataset["submissions"].items():
            pdf_file = os.path.join(
                pdf_directory, submission["content"]["paperhash"]["value"] + ".pdf"
            )
            paperhash = submission["content"]["paperhash"]["value"]
            if os.path.exists(pdf_file):
                paperhashes.append(paperhash)
                pdf_files.append(pdf_file)

        parsed_pdfs = parse_pdfs(
            paperhashes,
            pdf_files,
            pdf_directory,
            config["grobid"]["max_workers"],
            remove=config["clean_up"],
        )

        for submission_id, submission in dataset["submissions"].items():
            paperhash = submission["content"]["paperhash"]["value"]
            if paperhash in parsed_pdfs:
                submission["content"]["parsed_pdf"] = parsed_pdfs[paperhash]
            else:
                logging.warning(f"Could not find pdf for submission {submission_id}")
                submission["content"]["parsed_pdf"] = {}

    else:
        for submission_id, submission in dataset["submissions"].items():
            pdf_path = os.path.join(
                config["save_directory"],
                "pdfs",
                f"{submission['content']['paperhash']['value']}.pdf",
            )
            if not os.path.exists(pdf_path):
                logging.warning(f"Could not find pdf for submission {submission_id}")
                continue

            # Parse pdf
            parsed_pdf = parse_pdf(pdf_path, pdf_directory, remove=config["clean_up"])
            submission["content"]["parsed_pdf"] = parsed_pdf
    end = time.time()
    logging.info(f"Parsing pdfs took {end - start} seconds")