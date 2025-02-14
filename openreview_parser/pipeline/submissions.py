"""Obtaining the submissions from the OpenReview data model and parse them into the data model."""

import json
import logging
import os

from datetime import datetime

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
from openreview_parser.scientific_databases.openreview import (
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
    Reference,
)

SPECIAL_CASES_DECISIONS = ["ICLR_cc_2024_Conference", "ICLR_cc_2025_Conference"]


def submissions2papers(
    venue_instances: set[VenueInstance],
    openreview_client: openreview.Client,
    config: dict,
) -> None:
    """
    Process submissions and convert them into paper objects.

    Args:
        venue_instances (list[VenueInstance]): List of VenueInstance objects representing the venues to process.
        openreview_client (openreview.Client): OpenReview client for accessing the OpenReview API.
        config (dict): Configuration settings for the processing pipeline.

    Returns:
        None
    """
    parsed_venues = load_existing_venue_datasets(config)

    if config["metadata"]["get_references"]:
        filepath = os.path.join(config["s2_data_directory"], "all_titles.json")
        assert os.path.exists(
            filepath
        ), f"File all_s2_titles.json not found in data directory"
        with open(filepath, "r") as file:
            all_s2_titles = json.load(file)

        filepath = os.path.join(config["s2_data_directory"], "key_dictionary.json")
        assert os.path.exists(
            filepath
        ), f"File key2file.json not found in data directory"
        with open(filepath, "r") as file:
            key2file = json.load(file)

    (
        note_type_mapping,
        submission_fields_mapping,
        review_fields_mapping,
        venue2review_encodings,
        decision_fields_mapping,
        venue2decision_encodings,
    ) = load_encodings()

    total_n_venues_to_process = len(venue_instances)
    for i, venue_instance in enumerate(venue_instances):
        print(f"Parsing {venue_instance} ({i/total_n_venues_to_process*100:.2f}%)")
        venue_id = venue_instance.venue.replace("/", "_").replace(".", "_")

        if venue_id in parsed_venues:
            logging.info(f"Skipping {venue_instance.venue} as it already exists")
            continue

        logging.info(f"Processing {venue_instance.venue}")
        submissions = get_submissions(
            openreview_client, venue_instance, config["use_openreview_api_v2"]
        )

        if len(submissions) == 0:
            logging.info(f"No submissions found for {venue_instance.venue}")
            continue

        statistics = parse_submissions(
            submissions,
            note_type_mapping,
            submission_fields_mapping,
            review_fields_mapping,
            venue2review_encodings,
            decision_fields_mapping,
            venue2decision_encodings,
            all_s2_titles,
            key2file,
            openreview_client,
            venue_id,
            config,
        )
        if statistics is None:
            logging.warning(f"Could not parse submissions for {venue_instance.venue}")
        else:
            n_submissions, n_reviews = statistics
            logging.info(f"Finished processing {venue_id} submissions")
            logging.info(
                f"Retrieved {n_submissions} submissions and {n_reviews} reviews"
            )


def load_encodings() -> tuple[dict, dict, dict, dict, dict, dict]:
    """
    Load the encodings for the review fields.

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
    """
    Load existing venue datasets from a JSON file.

    Args:
        config (dict): Configuration dictionary containing the save directory.

    Returns:
        list[str]: List of parsed venue instances.
    """
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
    all_s2_titles: set,
    key2file: dict,
    client: OpenReviewClient,
    venue_id: str,
    config: dict,
) -> tuple[int, int] | None:
    """
    Parse a list of OpenReview submissions and extracts relevant information.

    Args:
        submissions (list[openreview.Note]): List of OpenReview submission notes.
        note_type_mapping (dict): Mapping of note types to their corresponding labels.
        submission_fields_mapping (dict): Mapping of submission fields to their corresponding labels.
        review_fields_mapping (dict): Mapping of review fields to their corresponding labels.
        venue2review_encodings (dict): Mapping of venue IDs to review field encodings.
        decision_fields_mapping (dict): Mapping of decision fields to their corresponding labels.
        venue2decision_encodings (dict): Mapping of venue IDs to decision field encodings.
        all_s2_titles (set): Set of all S2 titles.
        key2file (dict): Mapping of keys to file paths.
        client (OpenReviewClient): OpenReview client object.
        venue_id (str): ID of the venue.
        config (dict): Configuration settings.

    Returns:
        tuple[int, int] | None: A tuple containing the number of submissions and the number of reviews processed, or None if an error occurred.
    """
    if config["metadata"]["classify_sections"]:
        section_classifier = SectionClassifier.load_from_checkpoint(
            "./model_store/section_classifier_openreview.ckpt",
            map_location=config["device"],
        )
        section_classifier.load_preprocessing_utils(config["device"])

    n_submissions, n_reviews = 0, 0
    for submission in tqdm.tqdm(submissions[:2]):
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

        if config["use_openreview_api_v2"]:
            paperhash = (
                submission["content"]["paperhash"]["value"].replace("/", "_")
                + "|"
                + venue_id
            )
        else:
            paperhash = (
                submission["content"]["paperhash"].replace("/", "_") + "|" + venue_id
            )

        if os.path.exists(
            os.path.join(config["save_directory"], "papers", f"{paperhash}.json")
        ):
            logging.info(f"Skipping {paperhash} as it already exists")
            continue

        paper_info["paperhash"] = paperhash
        paper_info["venue"] = venue_id
        paper_info["publication_date"] = get_publication_date(submission)
        paper_info["license"] = submission.get("license", None)

        # Parse submission
        submission_info = parse_submission_note(
            submission, submission_fields_mapping, config["use_openreview_api_v2"]
        )
        paper_info.update(submission_info)

        if paper_info is None:
            logging.info(f"Could not parse submission {paperhash} from {venue_id}")
            continue

        # Get reviews and comments and meta_review
        notes = get_notes(client, forum=forum_id)
        reviews, comments = [], []

        decision = None
        for note in notes:
            if config["use_openreview_api_v2"]:
                note_type = note.invitations[0].split("/")[-1]
            else:
                note_type = note.invitation.split("/")[-1]

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
        comments = process_comments(comments, api_v2=config["use_openreview_api_v2"])
        review_fields_encoding = venue2review_encodings[venue_id]
        reviews = process_reviews(
            reviews,
            review_fields_mapping,
            review_fields_encoding,
            config["use_openreview_api_v2"],
        )
        decision_encodings = venue2decision_encodings[venue_id]

        if decision is None:
            if venue_id in SPECIAL_CASES_DECISIONS:
                decision, decision_text = handle_special_cases_decision(
                    submission, venue_id
                )
            if decision is None:
                logging.info(
                    f"Could not find decision for submission {id} ({venue_id})"
                )
                decision = None
                decision_text = None
        else:
            decision, decision_text = process_decision(
                decision,
                decision_fields_mapping,
                decision_encodings,
                venue_id,
                config["use_openreview_api_v2"],
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
        paper.create_bibref2paperhash()
        paper.n_references = len(paper.bibref2paperhash)

        # Section classifier
        if config["metadata"]["classify_sections"]:
            paper = classify_sections(paper, section_classifier, config)
            paper.create_bibref2section()

        # Get citation score
        if config["metadata"]["get_s2_info"]:
            # Obtain citation counts for accepted papers
            if paper.decision:
                s2_info = get_s2info(
                    paper.title,
                    ["citationCount", "influentialCitationCount", "externalIds"],
                    config["use_s2_api_key"],
                )
                if s2_info is None:
                    logging.warning(f"Could not get s2 info for submission {id}")
                elif len(s2_info) != 0:
                    paper.n_citations = s2_info.get("citationCount", None)
                    paper.n_influential_citations = s2_info.get(
                        "influentialCitationCount", None
                    )
                    paper.external_ids = s2_info.get("externalIds", None)
                    if paper.external_ids is not None:
                        paper.s2_corpus_id = paper.external_ids.get("CorpusId", None)
                        paper.arxiv_id = paper.external_ids.get("ArXiv", None)

        # Get references
        if config["metadata"]["get_references"]:
            # For accepted papers get references from s2
            if paper.decision and paper.s2_corpus_id is not None:
                references = get_s2_references(
                    paper.s2_corpus_id, "corpus_id", config["use_s2_api_key"]
                )
            else:
                # For rejected papers get references from the GROBID parse
                references = get_references_grobid(
                    paper, all_s2_titles, key2file, config
                )

            paper.references = references

        # Get hypotheses
        if config["metadata"]["annotate_hypotheses"]:
            hypothesis = annotate_paper(paper, config["llm"])
            paper.hypothesis = hypothesis

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
    """
    Handle special cases for decision handling.

    Args:
        submission (dict): The submission dictionary.
        venue_id (str): The ID of the venue.

    Returns:
        tuple[bool | None, str | None]: A tuple containing a boolean value and a string value.
            The boolean value indicates the decision value.
            The string value provides the decision text.
    """
    if venue_id == "ICLR_cc_2024_Conference":
        if "withdrawn" in submission["content"]["venue"]["value"].lower():
            return False, "Withdrawn"

    if venue_id == "ICLR_cc_2025_Conference":
        if "withdrawn" in submission["content"]["venue"]["value"].lower():
            return False, "Withdrawn"

    return None, None


def parse_submission_note(
    submission: dict, submission_fields_mapping: dict, api_v2: bool = True
) -> dict:
    """
    Parse a submission note based on the provided submission and submission_fields_mapping.

    Args:
        submission (dict): The submission note to be parsed.
        submission_fields_mapping (dict): A dictionary mapping the fields of the submission note to their corresponding keys.
        api_v2 (bool, optional): Flag indicating whether to use API v2. Defaults to True.

    Returns:
        dict: The parsed submission note.

    Raises:
        None
    """
    if api_v2:
        return parse_submission_note_v2(submission, submission_fields_mapping)
    else:
        return parse_submission_note_v1(submission, submission_fields_mapping)


def parse_submission_note_v1(submission: dict, submission_fields_mapping: dict) -> dict:
    """
    Parse a submission note and extracts relevant information based on the provided fields mapping.

    Args:
        submission (dict): The submission note to be parsed.
        submission_fields_mapping (dict): A dictionary mapping the fields in the submission note to the corresponding model fields.

    Returns:
        dict: A dictionary containing the extracted information from the submission note.
    """
    paper_info: dict = {}
    for key, value in submission["content"].items():
        if key not in submission_fields_mapping:
            continue

        model_field = submission_fields_mapping[key]
        if model_field == None:
            continue

        elif model_field == "field_of_study":
            # Check whether its str or list; convert str to list[str]
            if isinstance(value, str):
                field_of_study_value = [value]
            else:
                field_of_study_value = value

            # Check whether field already exists, if yes concatenate
            if "field_of_study" in paper_info:
                paper_info["field_of_study"] += field_of_study_value
            else:
                paper_info["field_of_study"] = field_of_study_value
        elif model_field == "authors":
            if isinstance(value, str):
                paper_info["authors"] = [value]
            else:
                paper_info["authors"] = value
        else:
            paper_info[model_field] = value

    return paper_info


def parse_submission_note_v2(submission: dict, submission_fields_mapping: dict) -> dict:
    """
    Parse a submission note and extracts relevant information based on the provided submission fields mapping.

    Args:
        submission (dict): The submission note to be parsed.
        submission_fields_mapping (dict): A dictionary mapping submission fields to model fields.

    Returns:
        dict: A dictionary containing the extracted information from the submission note.
    """
    paper_info: dict = {}
    for key, value in submission["content"].items():
        if key not in submission_fields_mapping:
            continue

        model_field = submission_fields_mapping[key]
        if model_field == None:
            continue

        elif model_field == "field_of_study":
            # Check whether its str or list; convert str to list[str]
            if isinstance(value["value"], str):
                field_of_study_value = [value["value"]]
            else:
                field_of_study_value = value["value"]

            # Check whether field already exists, if yes concatenate
            if "field_of_study" in paper_info:
                paper_info["field_of_study"] += field_of_study_value
            else:
                paper_info["field_of_study"] = field_of_study_value
        elif model_field == "authors":
            if isinstance(value["value"], str):
                paper_info["authors"] = [value["value"]]
            else:
                paper_info["authors"] = value["value"]
        else:
            paper_info[model_field] = value["value"]

    return paper_info


def parse_pdf(
    paperhash: str,
    max_workers: int,
    config: dict,
    remove: bool = False,
) -> dict[str, dict] | None:
    """
    Parse a PDF file and convert it to a JSON representation using GROBID.

    Args:
        paperhash (str): The hash of the paper.
        max_workers (int): The maximum number of workers to use for processing.
        config (dict): The configuration dictionary.
        remove (bool, optional): Whether to remove the PDF and XML files after parsing. Defaults to False.

    Returns:
        dict[str, dict] | None: The parsed paper as a JSON representation, or None if parsing failed.
    """
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
        logging.warning(f"Parsing of the pdf failed, no xml file found for {paperhash}")
        return None

    paper = convert_tei_xml_file_to_s2orc_json(xml_file)
    paper = paper.release_json()

    if remove:
        if os.path.exists(pdf_file):
            os.remove(pdf_file)

        if os.path.exists(xml_file):
            os.remove(xml_file)

    return paper


def process_comments(comments: list[dict], api_v2: bool = True) -> list[Comment]:
    """
    Process comments based on the specified API version.

    Args:
        comments (list[dict]): A list of comment dictionaries.
        api_v2 (bool, optional): Flag indicating whether to use API v2. Defaults to True.

    Returns:
        list[Comment]: A list of processed Comment objects.
    """
    if api_v2:
        return process_comments_v2(comments)
    else:
        return process_comments_v1(comments)


def process_comments_v1(comments: list[dict]) -> list[Comment]:
    """
    Postprocesse a list of comments and returns a list of Comment objects.

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

        comment_text = comment
        if title != None:
            title = title

        processed_comments.append(Comment(title=title, comment=comment_text))

    return processed_comments


def process_comments_v2(comments: list[dict]) -> list[Comment]:
    """
    Postprocesse a list of comments and returns a list of Comment objects.

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

        comment_text = comment["value"]
        if title != None:
            title = title["value"]

        processed_comments.append(Comment(title=title, comment=comment_text))

    return processed_comments


def get_publication_date(submission_note: openreview.Note) -> str | None:
    """
    Get the publication date of a submission note.

    Args:
        submission_note (openreview.Note): The submission note.

    Returns:
        str | None: The publication date in the format "YYYY-MM-DD" or None if not available.
    """
    p_date = submission_note["content"].get("publication_date", None)
    if p_date is None:
        # odate contains the Unix timestamp in miliseconds when Note becomes public
        p_date = submission_note.get("odate", None)
        if p_date is not None:
            p_date = p_date / 1000
            p_date = datetime.fromtimestamp(p_date).strftime("%Y-%m-%d")
    else:
        if isinstance(p_date, str):
            p_date = p_date
        else:
            p_date = p_date["value"]

    return p_date


def process_reviews(
    reviews: list[dict],
    review_field_mapping: dict,
    review_encodings: dict,
    api_v2: bool = True,
) -> list[Review]:
    """
    Process a list of reviews and convert them into a list of processed reviews.

    Args:
        reviews (list[dict]): A list of review dictionaries.
        review_field_mapping (dict): A dictionary mapping review fields to their types.
        review_encodings (dict): A dictionary containing encodings for specific review fields.
        api_v2 (bool, optional): A flag indicating whether the reviews are in API v2 format. Defaults to True.

    Returns:
        list[Review]: A list of processed Review objects.
    """
    processed_reviews = []

    for review in reviews:
        encoded_review = {"review_id": review["id"]}
        text_review = {}
        for field, field_value in review["content"].items():
            if api_v2:
                field_value = str(field_value["value"])
            else:
                field_value = str(field_value)

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

        parsed_text_review = TextReview(**text_review)
        encoded_review["review"] = parsed_text_review

        parsed_review = Review(**encoded_review)
        processed_reviews.append(parsed_review)

    return processed_reviews


def process_decision(
    decision: dict,
    decision_mapping: dict,
    decision_encodings: dict,
    venue_id: str,
    api_v2: bool = True,
) -> tuple[bool | None, str | None]:
    """
    Process the decision information for a submission.

    Args:
        decision (dict): The decision information for a submission.
        decision_mapping (dict): A mapping of decision keys to model fields.
        decision_encodings (dict): A mapping of decision values to encoded values.
        venue_id (str): The ID of the venue.
        api_v2 (bool, optional): Whether to use the API v2 format. Defaults to True.

    Returns:
        tuple[bool | None, str | None]: A tuple containing the boolean decision value and the decision text.
    """
    decision_text = ""
    decision_bool = None
    for key, value in decision["content"].items():
        if api_v2:
            value = value["value"]
        else:
            value = value

        if key not in decision_mapping:
            logging.warning(f"Key {key} not in decision mapping for {venue_id}.")
            continue
        model_field = decision_mapping[key]

        if model_field == "decision":
            # For some venues, the decision is a list
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                value = value[0]
            if value not in decision_encodings[model_field]:
                logging.info(f"Value {value} not in decision encodings for {venue_id}.")
                continue
            decision_bool = decision_encodings[model_field][value]
        elif model_field == "decision_text":
            decision_text += value
        else:
            continue

    return decision_bool, decision_text


def get_references_grobid(
    paper: Paper, all_s2_titles: set, key2file: dict, config: dict
) -> list[Reference]:
    """
    Retrieve references from GROBID parse.

    Args:
        paper (Paper): The paper object.
        all_s2_titles (set): Set of all S2 titles.
        key2file (dict): Dictionary mapping keys to file names.
        config (dict): Configuration dictionary.

    Returns:
        list[Reference]: List of references.
    """
    references = []

    if not (
        paper.parsed_pdf is None
        or "pdf_parse" not in paper.parsed_pdf
        or paper.parsed_pdf["pdf_parse"] is None
        or "bib_entries" not in paper.parsed_pdf["pdf_parse"]
    ):
        for value in paper.parsed_pdf["pdf_parse"]["bib_entries"].values():
            title = value["title"].strip().lower()

            if title in all_s2_titles:
                # Find the file to open
                file = find_file(title, key2file)
                filepath = os.path.join(
                    config["s2_data_directory"], "parsed_paper_info", f"{file}.json"
                )
                with open(filepath, "r") as f:
                    title2s2info = json.load(f)
                    s2info = title2s2info[title]

                    external_ids = s2info.get("externalIds", None)
                    if external_ids is not None:
                        corpus_id = external_ids.get("CorpusId", None)
                        if corpus_id is not None:
                            corpus_id = str(corpus_id)
                        arxiv_id = external_ids.get("ArXiv", None)
                    else:
                        corpus_id = None
                        arxiv_id = None

                    abstract = s2info.get("abstract", "")
                    if abstract is None:
                        abstract = ""

                    reference_info = {
                        "title": title,
                        "abstract": abstract,
                        "authors": s2info.get("authors", []),
                        "external_ids": external_ids,
                        "arxiv_id": arxiv_id,
                        "s2_corpus_id": corpus_id,
                    }

                    reference = Reference(**s2info)
                    references.append(reference)

            else:
                s2info = get_s2info(
                    title,
                    ["title", "abstract", "authors", "externalIds"],
                    config["use_s2_api_key"],
                )

                if s2info is None:
                    logging.info(f"Could not get s2 info for reference {title}")
                    continue

                external_ids = s2info.get("externalIds", None)
                if external_ids is not None:
                    corpus_id = external_ids.get("CorpusId", None)
                    if corpus_id is not None:
                        corpus_id = str(corpus_id)
                    arxiv_id = external_ids.get("ArXiv", None)
                else:
                    corpus_id = None
                    arxiv_id = None

                authors = s2info.get("authors", [])
                authors = [author["name"] for author in authors]

                abstract = s2info.get("abstract", "")
                if abstract is None:
                    abstract = ""

                reference_info = {
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "arxiv_id": arxiv_id,
                    "s2_corpus_id": corpus_id,
                    "external_ids": external_ids,
                }

                reference = Reference(**reference_info)
                references.append(reference)

    return references


def find_file(title: str, key2file: dict) -> str | None:
    """
    Find the file key associated with a given title within a dictionary of file keys and intervals.

    Parameters:
    - title (int): The title to search for.
    - key2file (dict): A dictionary mapping file keys to intervals.

    Returns:
    - str or None: The file key associated with the given title, or None if no interval is found.
    """
    intervals = [(start, end, filekey) for filekey, (start, end) in key2file.items()]

    left, right = 0, len(intervals) - 1

    while left <= right:
        mid = (left + right) // 2
        start, end, _ = intervals[mid]

        if start <= title <= end:
            return intervals[mid][-1]  # Element is in this interval
        elif title < start:
            right = mid - 1  # Search in the left half
        else:
            left = mid + 1  # Search in the right half

    return None  # No interval found
