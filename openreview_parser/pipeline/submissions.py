"""
Obtaining the submissions from the OpenReview data model and parse them into the data model.
"""

import json
import logging
import os


import openreview
import tqdm

from doc2json.grobid2json.grobid.grobid_client import (
    GrobidClient,
)
from doc2json.grobid2json.tei_to_json import (
    convert_tei_xml_file_to_s2orc_json,
)

from openreview.api import OpenReviewClient


from openreview_parser.hypothesis_annotation.annotate import annotate_paper
from openreview_parser.pipeline.utils import urlretrieve_backoff
from openreview_parser.scientific_databases.openreview_v2 import (
    get_submissions,
    get_notes,
)
from openreview_parser.scientific_databases.s2 import get_s2info, get_s2_references

from openreview_parser.section_classification.classify import (
    classify_sections,
    SectionClassifier,
)

from openreview_parser.utils.data import (
    VenueInstance,
    Comment,
    Review,
    TextReview,
    Paper,
)

SPECIAL_CASES_DECISIONS = ["ICLR_cc_2024_Conference", "ICLR_cc_2025_Conference"]


def submissions2papers(
    venue_instances: list[VenueInstance],
    openreview_client: openreview.Client,
    config: dict,
) -> None:
    parsed_venues = load_existing_venue_datasets(config)

    (
        note_type_mapping,
        submission_fields_mapping,
        review_fields_mapping,
        venue2review_encodings,
        decision_fields_mapping,
        venue2decision_encodings,
    ) = load_encodings(config)

    for venue_instance in venue_instances:
        print(venue_instance)
        venue_id = venue_instance.venue.replace("/", "_").replace(".", "_")

        if venue_id in parsed_venues:
            logging.info(f"Skipping {venue_instance.venue} as it already exists")
            continue

        logging.info(f"Processing {venue_instance.venue}")
        submissions = get_submissions(openreview_client, venue_instance)

        if len(submissions) == 0:
            logging.info(f"No submissions found for {venue_instance.venue}")
            continue

        n_submissions, n_reviews = parse_submissions(
            submissions,
            note_type_mapping,
            submission_fields_mapping,
            review_fields_mapping,
            venue2review_encodings,
            decision_fields_mapping,
            venue2decision_encodings,
            openreview_client,
            venue_id,
            config,
        )

        logging.info(f"Finished processing {venue_id} submissions")
        logging.info(f"Retrieved {n_submissions} submissions and {n_reviews} reviews")


def load_encodings(config: dict) -> dict:
    """
    Loads the encodings for the review fields.

    Args:
        config (dict): The configuration settings.

    Returns:
        dict: The mapping of note types to their corresponding labels.
    """

    directory = "./data"
    with open(os.path.join(directory, "note_type_mapping.json")) as file:
        note_type_mapping = json.load(file)

    with open(os.path.join(directory, "review_fields_mapping.json")) as file:
        review_fields_mapping = json.load(file)

    with open(os.path.join(directory, "venue2review_encodings.json")) as file:
        venue2review_encodings = json.load(file)

    with open(os.path.join(directory, "decision_fields_mapping.json")) as file:
        decision_fields_mapping = json.load(file)

    with open(os.path.join(directory, "venue2decision_encodings.json")) as file:
        venue2decision_encodings = json.load(file)

    with open(os.path.join(directory, "submission_fields_mapping.json")) as file:
        submission_fields_mapping = json.load(file)

    return (
        note_type_mapping,
        submission_fields_mapping,
        review_fields_mapping,
        venue2review_encodings,
        decision_fields_mapping,
        venue2decision_encodings,
    )


def load_existing_venue_datasets(config: dict) -> list[str]:
    parsed_venues_file = os.path.join(
        config["save_directory"], "venues", "parsed_venues.json"
    )
    if os.path.exists(parsed_venues_file):
        with open(
            os.path.join(config["save_directory"], "venues", "parsed_venues.json"), "r"
        ) as file:
            parsed_venues = json.load(file)
            parsed_venues = [VenueInstance(**vi) for vi in parsed_venues]
    else:
        parsed_venues = []

    return parsed_venues


def parse_submissions(
    submissions: list[openreview.Note],
    note_type_mapping: dict,
    submission_fields_mapping: dict,
    review_fields_mapping: dict,
    venue2review_encodings: dict,
    decision_fields_mapping: dict,
    venue2decision_encodings: dict,
    client: OpenReviewClient,
    venue_id: str,
    config: dict,
) -> tuple[dict, int]:
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

    if config["metadata"]["classify_sections"]:
        section_classifier = SectionClassifier.load_from_checkpoint(
            "./model_store/section_classifier_openreview.ckpt",
            map_location=config["device"],
        )
        section_classifier.load_preprocessing_utils(config["device"])

    n_submissions, n_reviews = 0, 0
    for submission in tqdm.tqdm(submissions):
        paper_info = {}
        submission = submission.to_json()
        id = submission["id"]
        forum_id = submission["forum"]

        # Obtain and check paper id existence
        if "paperhash" not in submission["content"]:
            logging.info(
                f"Submission {id} in venue {venue_id} does not have a paperhash."
            )
            return None

        paperhash = submission["content"]["paperhash"]["value"].replace("/", "_")

        if os.path.exists(
            os.path.join(config["save_directory"], "papers", f"{paperhash}.json")
        ):
            logging.info(f"Skipping {paperhash} as it already exists")
            continue

        paper_info["paperhash"] = paperhash

        # Parse submission
        submission_info = parse_submission_note(submission, submission_fields_mapping)
        paper_info.update(submission_info)

        if paper_info is None:
            logging.info(f"Could not parse submission {paperhash} from {venue_id}")
            continue

        # Get reviews and comments and meta_review
        notes = get_notes(client, forum=forum_id)
        reviews, comments = [], []

        decision = None
        for note in notes:
            note_type = note.invitations[0].split("/")[-1]
            print(note_type)
            if note_type == "Blind_Submission":
                continue
            if note_type not in note_type_mapping:
                logging.warning(
                    f"Trying to parse an unknown Note Type: {note_type} for submission: {id} at venue {venue_id}"
                )
                continue
            note_type = note_type_mapping[note_type]

            if note_type == "Review":
                reviews.append(note.to_json())

            if note_type == "Comment":
                comments.append(note.to_json())

            if note_type == "Decision":
                decision = note.to_json()

        if decision == None:
            if venue_id in SPECIAL_CASES_DECISIONS:
                decision,
        comments = process_comments(comments)
        review_fields_encoding = venue2review_encodings[venue_id]
        reviews = process_reviews(
            reviews, review_fields_mapping, review_fields_encoding
        )
        decision_encodings = venue2decision_encodings[venue_id]

        if decision is None:
            if venue_id in SPECIAL_CASES_DECISIONS:
                decision, decision_text = handle_special_cases_decision(
                    submission, venue_id
                )
            if decision is None:
                logging.info(f"Could not find decision for submission {id}")
        else:
            decision, decision_text = process_decision(
                decision, decision_fields_mapping, decision_encodings, venue_id
            )

        paper_info["reviews"] = reviews
        paper_info["comments"] = comments
        paper_info["decision"] = decision
        paper_info["decision_text"] = decision_text

        """
        For each submission do the following steps:
        1. Parse the submission PDF 
        2. Classify the section 
        3. Retrieve the referecences 
        4. Label if with hypotheses 
        5. Obtain citation count if accepted 
        """

        # Get pdf
        url = f"https://openreview.net/pdf?id={id}"
        filename = os.path.join(config["save_directory"], "pdfs", f"{paperhash}.pdf")
        response = urlretrieve_backoff(url, filename)
        print(response)
        if response is None:
            logging.info(f"Could not download pdf for submission {id}")
            continue

        parsed_pdf = parse_pdf(
            paperhash,
            config["grobid"]["max_workers"],
            config,
            remove=config["clean_up"],
        )
        paper_info["parsed_pdf"] = parsed_pdf

        # Create paper object and save it to json
        paper = Paper(
            **paper_info,
        )

        # Organize structured content
        paper.organize_text()

        # Section classifier
        if config["metadata"]["classify_sections"]:
            paper = classify_sections(paper, section_classifier, config)

        # Get citation score
        if config["metadata"]["get_s2_info"]:
            # Obtain citation counts for accepted papers
            if paper.decision:
                s2_info = get_s2info(
                    paper.title,
                    ["citationCount", "influentialCitationCount", "externalIds"],
                    config["use_s2_api_key"],
                )
                print(s2_info)
                if len(s2_info) != 0:
                    paper.n_citations = s2_info.get("citationCount", None)
                    paper.n_influential_citations = s2_info.get(
                        "influentialCitationCount", None
                    )
                    paper.external_ids = s2_info.get("externalIds", None)

        # Get references
        if config["get_references"]:
            # For accepted papers get references from s2
            if paper.decision and paper.s2_corpus_id is not None:
                references = get_s2_references(paper)
            else:
                # For rejected papers get references from the GROBID parse
                references = get_references_grobid(paper)

            paper.references = references

        # Get hypotheses
        if config["metadata"]["annotate_hypotheses"]:
            paper = annotate_paper(paper)

        with open(
            os.path.join(config["save_directory"], "papers", f"{paperhash}.json"), "w+"
        ) as file:
            json.dump(paper.model_dump(), file, indent=4)

        n_submissions += 1
        n_reviews += len(reviews)

    return n_submissions, n_reviews


def handle_special_cases_decision(
    submission: dict, venue_id: str
) -> tuple[bool | None, str | None]:
    if venue_id == "ICLR_cc_2024_Conference":
        if "withdrawn" in submission["content"]["venue"]["value"].lower():
            return False, "Withdrawn"
        else:
            return None, None


def parse_submission_note(submission: dict, submission_fields_mapping: dict) -> dict:
    paper_info = {}
    print(submission)
    for key, value in submission["content"].items():
        if key not in submission_fields_mapping:
            continue

        model_field = submission_fields_mapping[key]
        if model_field == None:
            continue

        elif model_field == "field_of_study":
            # Check whether its str or list; convert str to list[str]
            if isinstance(value["value"], str):
                paper_info[model_field] = [value["value"]]
            else:
                paper_info[model_field] = value["value"]
        else:
            paper_info[model_field] = value["value"]

    return paper_info


def parse_pdf(
    paperhash: str,
    max_workers: int,
    config: dict,
    remove: bool = False,
) -> dict[str, dict]:
    pdf_directory = os.path.join(config["save_directory"], "pdfs")
    pdf_file = os.path.join(pdf_directory, f"{paperhash}.pdf")
    xml_directory = os.path.join(config["save_directory"], "xmls")
    xml_file = os.path.join(xml_directory, f"{paperhash}.tei.xml")

    # Dict to store parsed papers
    papers: dict[str, dict] = {}

    # GROBID client
    grobid_client = GrobidClient()
    grobid_client.max_workers = max_workers

    # PDF -> XML
    grobid_client.process_batch([pdf_file], xml_directory, "processFulltextDocument")

    # XML -> JSON
    if not os.path.exists(xml_file):
        return None

    paper = convert_tei_xml_file_to_s2orc_json(xml_file)
    paper = paper.release_json()

    if remove:
        if os.path.exists(pdf_file):
            os.remove(pdf_file)

        if os.path.exists(xml_file):
            os.remove(xml_file)

    return paper


def process_comments(comments: list[dict]) -> list[Comment]:
    """
    Postprocesses a list of comments and returns a list of Comment objects.

    Args:
        comments (list[dict]): The list of comments to be processed.

    Returns:
        list[Comment]: The processed list of Comment objects.
    """
    processed_comments = []
    for comment in comments:
        content = comment["content"]
        title = content.get("title", None)
        comment = content.get("comment", None)

        if comment == None:
            continue

        if comment != None:
            comment = comment["value"]
        if title != None:
            title = title["value"]

        processed_comments.append(Comment(title=title, comment=comment))

    return processed_comments


def process_reviews(
    reviews: list[dict],
    review_field_mapping: dict,
    review_encodings: dict,
) -> list[Review]:
    """
    Postprocesses a list of reviews.

    Args:
        reviews (list[dict]): The list of reviews to be postprocessed.
        dataset_name (str): The name of the dataset.
        directory (str): The directory path.

    Returns:
        list[Review]: The processed reviews.
    """
    processed_reviews = []

    for review in reviews:
        encoded_review = {"review_id": review["id"]}
        text_review = {}
        for field, field_value in review["content"].items():
            field_value = str(field_value["value"])
            field_type = review_field_mapping[field]

            if field_type == None:
                continue

            if (
                field_type
                in [
                    "score",
                    "confidence",
                    "novelty",
                    "correctness",
                    "clarity",
                    "impact",
                    "reproducibility",
                ]
                and field in review_encodings
            ):
                encoding = review_encodings[field]
                if field_value not in encoding:
                    continue

                field_value = encoding[field_value]
                encoded_review[field_type] = float(field_value)

            elif "review" in field_type:
                review_part = field_type.split("|")[1]
                text_review[review_part] = f"{review_part}: {field_value}"

        text_review = TextReview(**text_review)
        encoded_review["review"] = text_review

        review = Review(**encoded_review)
        processed_reviews.append(review)

    return processed_reviews


def process_decision(
    decision: dict, decision_mapping: dict, decision_encodings: dict, venue_id: str
) -> tuple[bool | None, str | None]:
    """
    Postprocesses a decision and returns the decision and decision text.

    Args:
        decision (dict): The decision to be processed.
        dataset_name (str): The name of the dataset.
        directory (str): The directory path.

    Returns:
        tuple[str, str]: The decision and decision text.
    """

    decision_text = ""
    decision_bool = None
    for key, value in decision["content"].items():
        value = value["value"]
        if key not in decision_mapping:
            logging.warning(f"Key {key} not in decision mapping for {venue_id}.")
            continue
        model_field = decision_mapping[key]

        if model_field == "decision":
            if value not in decision_encodings[model_field]:
                logging.info(f"Value {value} not in decision encodings for {venue_id}.")
                continue
            decision_bool = decision_encodings[model_field][value]
        elif model_field == "decision_text":
            decision_text += value
        else:
            continue

    return decision_bool, decision_text


def get_references_grobid():
    pass
