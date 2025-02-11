"""
Utilities to build schemas for the venue instances
"""

import glob
import json
import logging
import os

from collections import defaultdict
from typing import Union

import numpy as np
import openreview

from openreview.api import OpenReviewClient
from tqdm import tqdm

from openreview_parser_v2.scientific_databases.openreview import get_submissions
from openreview_parser_v2.utils.data import VenueInstance

REVIEW_FIELDS = [
    "score",
    "confidence",
    "novelty",
    "correctness",
    "clarity",
    "impact",
    "ethics",
    "reproducibility",
    "review|paper_summary",
    "review|main_review",
    "review|questions",
    "review|title",
    "review|strength_weakness",
    "review|limitations",
    "",
]

DECISION_FIELDS = ["decision", "decision_text", ""]

SUBMISSION_FIELDS = [
    "authors",
    "title",
    "abstract",
    "publication_data",
    "summary",
    "field_of_study",
    "",
]


def get_schemas(
    client: OpenReviewClient, venues: list[VenueInstance], config: dict
) -> list[VenueInstance]:
    """
    Retrieves and processes schemas for the given venues.

    Args:
        client (OpenReviewClient): An instance of the OpenReviewClient class.
        venues (list[VenueInstance]): A list of VenueInstance objects representing the venues.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        list[VenueInstance]: A list of VenueInstance objects with updated schemas.

    """
    existing_schemas, venues_with_infos = load_schema_data(config)
    (
        note_type_mapping,
        submission_fields_mapping,
        review_fields_mapping,
        decision_fields_mapping,
    ) = load_encodings()

    for venue in tqdm(venues):
        print(type(venue))
        schemas: dict[str, Union[dict, list]] = {}

        # Check whether venue has already got a schema
        venue_id = venue.venue.replace("/", "_").replace(".", "_")

        if venue_id in existing_schemas:
            continue

        # Get submissions
        submissions = get_submissions(client, venue)

        if len(submissions) == 0:
            logging.warning(f"No submissions for Venue {venue.venue} found.")
            save_empty_schema(venue_id, config)
            continue

        note_type_mapping = update_note_types(submissions, note_type_mapping)
        submission_schema = get_submission_schema(submissions)
        submission_fields_mapping = update_submission_fields_mapping(
            submission_schema, submission_fields_mapping
        )
        reply_types = get_reply_types(submissions)
        review_schema = get_note_schema(submissions, note_type_mapping, "Review")
        review_fields_mapping = update_review_fields_mapping(
            review_schema, review_fields_mapping
        )
        comment_schema = get_note_schema(submissions, note_type_mapping, "Comment")
        decision_schema = get_note_schema(submissions, note_type_mapping, "Decision")
        decision_fields_mapping = update_decision_fields_mapping(
            decision_schema, decision_fields_mapping
        )

        if len(review_schema) == 0:
            logging.warning(f"No reviews for Venue {venue.venue} found.")
            save_empty_schema(venue_id, config)
            continue

        venues_with_infos.append(venue)

        schemas["submission_schema"] = submission_schema
        schemas["review_schema"] = review_schema
        schemas["comment_schema"] = comment_schema
        schemas["decision_schema"] = decision_schema
        schemas["reply_types"] = list(reply_types)

        # Map venue schemas to data models
        schemas2data_models(
            schemas, venue_id, review_fields_mapping, decision_fields_mapping
        )

        # Save info
        with open(
            os.path.join(config["save_directory"], "schemas", f"{venue_id}.json"), "w+"
        ) as file:
            json.dump(schemas, file, indent=4)

        with open(
            os.path.join(
                config["save_directory"], "venues", "filtered_venue_datasets.json"
            ),
            "w+",
        ) as file:
            json.dump(
                [venue.model_dump() for venue in venues_with_infos], file, indent=4
            )

    return venues_with_infos


def load_schema_data(config: dict) -> tuple[list[str], list[VenueInstance]]:
    """
    Load schema data from the specified directory.

    Args:
        save_directory (str): The directory path where the schema data is saved.

    Returns:
        tuple: A tuple containing the following:
            - existing schemas (list[str]): A list of existing schemas.
            - venues_with_infos (list[str]): A list of venues with submissions and reviews.
    """

    # Get existing schemas
    schema_directory = os.path.join(config["save_directory"], "schemas")
    schema_files = glob.glob(schema_directory + "/*.json")
    existing_schemas = [os.path.basename(file).split(".")[0] for file in schema_files]

    # Get venue strings of venues with submissions and reviews
    venues_with_infos = []
    filtered_venue_dataset_filepath = os.path.join(
        config["save_directory"], "venues", "filtered_venue_datasets.json"
    )
    if os.path.exists(filtered_venue_dataset_filepath):
        with open(filtered_venue_dataset_filepath, "r") as file:
            venues_with_infos = json.load(file)
            venues_with_infos = [VenueInstance(**vi) for vi in venues_with_infos]

    return existing_schemas, venues_with_infos


def load_encodings() -> tuple[dict, dict, dict, dict]:
    """
    Load encodings from JSON files.

    Returns:
        A tuple containing the encodings for note types, submission fields, review fields, and decision fields.
    """
    directory = "./data/"

    # Get note-type mapping
    if os.path.exists(os.path.join(directory, "note_type_mapping.json")):
        with open(os.path.join(directory, "note_type_mapping.json"), "r") as file:
            note_type_mapping = json.load(file)
    else:
        note_type_mapping = {}
    logging.info(f"Number of note types: {len(note_type_mapping)}")

    # Get review-field mapping
    if os.path.exists(os.path.join(directory, "review_fields_mapping.json")):
        with open(os.path.join(directory, "review_fields_mapping.json"), "r") as file:
            review_fields_mapping = json.load(file)
    else:
        review_fields_mapping = {}

    if os.path.exists(os.path.join(directory, "decision_fields_mapping.json")):
        with open(os.path.join(directory, "decision_fields_mapping.json"), "r") as file:
            decision_fields_mapping = json.load(file)
    else:
        decision_fields_mapping = {}

    if os.path.exists(os.path.join(directory, "submission_fields_mapping.json")):
        with open(
            os.path.join(directory, "submission_fields_mapping.json"), "r"
        ) as file:
            submission_field_mapping = json.load(file)
    else:
        submission_field_mapping = {}

    return (
        note_type_mapping,
        submission_field_mapping,
        review_fields_mapping,
        decision_fields_mapping,
    )


def save_empty_schema(venue_id: str, config: dict) -> None:
    """
    Save an empty schema for a given venue.

    Args:
        venue_id (str): The ID of the venue.
        config (dict): The configuration dictionary.

    Returns:
        None
    """

    schema_directory = os.path.join(config["save_directory"], "schemas")

    schemas: dict[str, Union[dict, list]] = {}
    schemas["submission_schema"] = {}
    schemas["review_schema"] = {}
    schemas["comment_schema"] = {}
    schemas["decision_schema"] = {}
    schemas["reply_types"] = []

    with open(os.path.join(schema_directory, f"{venue_id}.json"), "w") as file:
        json.dump(schemas, file, indent=4)


def get_submission_schema(submissions: list[openreview.Note]) -> dict:
    """
    Generate a submission schema based on a list of OpenReview submissions.

    Args:
        submissions (list[openreview.Note]): A list of OpenReview submissions.

    Returns:
        dict: The generated submission schema.
    """
    submission_schema: dict[str, list] = defaultdict(list)
    example_submission = submissions[0]

    for key, value in example_submission.content.items():
        if "value" not in value:
            value = None
        else:
            value = value["value"]
            if isinstance(value, list):
                value = "+".join(value)

        submission_schema[key].append(value)

    return submission_schema


def get_reply_types(submissions: list[openreview.Note]) -> list[str]:
    """
    Get a list of unique reply types from a list of submissions.

    Args:
        submissions (list[openreview.Note]): A list of submission notes.

    Returns:
        list[str]: A list of unique reply types.
    """
    reply_types = set()
    for sub in submissions:
        replies = sub.details["directReplies"]

        for reply in replies:
            for invitation in reply["invitations"]:
                reply_type = invitation.split("/")[-1]
                reply_types.add(reply_type)

    return list(reply_types)


def get_note_schema(
    submissions: list[openreview.Note], note_type_mapping: dict, note_type: str
) -> dict:
    """
    Retrieves the schema of notes of a specified type from a list of submissions.

    Args:
        submissions (list[openreview.Note]): A list of submissions.
        note_type_mapping (dict): A dictionary mapping invitation types to note types.
        note_type (str): The note type to retrieve the schema for.

    Returns:
        dict: The schema of the notes, represented as a dictionary.

    """
    notes = []

    # For each submission get the notes of the specified type
    for submission in submissions:
        for reply in submission.details["directReplies"]:
            for invitation in reply["invitations"]:
                invitation_type = invitation.split("/")[-1]
                if invitation_type not in note_type_mapping:
                    continue
                if note_type_mapping[invitation_type] == note_type:
                    notes.append(reply)
                    break

    # From notes derive schema
    note_schema: dict[str, dict[str, list]] = defaultdict(dict)
    if len(notes) == 0:
        return note_schema
    else:
        for note in notes:
            for key, value in note["content"].items():
                if "values" not in note_schema[key]:
                    note_schema[key]["values"] = []

                if "value" not in value:
                    value = None
                else:
                    value = value["value"]
                    if isinstance(value, list):
                        value = "+".join(value)

                note_schema[key]["values"].append(value)

    note_schema[key]["values"] = list(set(note_schema[key]["values"]))

    return note_schema


def update_note_types(
    submissions: list[openreview.Note], note_type_mapping: dict
) -> dict:
    """
    Updates the note types in the given submissions based on the provided note type mapping.

    Args:
        submissions (list[openreview.Note]): A list of submissions to update.
        note_type_mapping (dict): A dictionary mapping note types to note data models.

    Returns:
        dict: The updated note type mapping.

    """
    for sub in submissions:
        for reply in sub.details["directReplies"]:
            for invitation in reply["invitations"]:
                note_type = invitation.split("/")[-1]
                if note_type not in note_type_mapping:
                    note_data_model_type = input(
                        f"Map the note type '{note_type}' to the note data model (Decision,Review,Comment). If it does not fit any type hit ENTER:"
                    )
                    if note_data_model_type:
                        note_type_mapping[note_type] = note_data_model_type
                    else:
                        note_type_mapping[note_type] = None

    # Save the update version
    with open(os.path.join("./data", "note_type_mapping.json"), "w+") as file:
        json.dump(note_type_mapping, file, indent=4)

    return note_type_mapping


def update_submission_fields_mapping(
    submission_schema: dict, submission_fields_mapping: dict
) -> dict:
    """
    Updates the submission fields mapping based on the given submission schema.

    Args:
        submission_schema (dict): The schema of the submission.
        submission_fields_mapping (dict): The current mapping of fields to the submission data model.

    Returns:
        dict: The updated submission fields mapping.

    """
    for field, content in submission_schema.items():
        if field not in submission_fields_mapping:
            not_correct_field = True
            while not_correct_field:
                input_text = f"Map the field {field} to the submission data model\n"
                input_text += f"Some example values: {content}"
                input_text += f"Possible fields are: {SUBMISSION_FIELDS}."
                submission_data_model_field = input(f"{input_text} \n {field}:")
                if submission_data_model_field in SUBMISSION_FIELDS:
                    not_correct_field = False
                else:
                    print(
                        f"Field {submission_data_model_field} is not a valid submission_field."
                    )

            if submission_data_model_field:
                submission_fields_mapping[field] = submission_data_model_field
            else:
                submission_fields_mapping[field] = None

    with open(os.path.join("./data", "submission_fields_mapping.json"), "w+") as file:
        json.dump(submission_fields_mapping, file, indent=4)

    return submission_fields_mapping


def update_review_fields_mapping(
    review_schema: dict, review_fields_mapping: dict
) -> dict:
    """
    Updates the review fields mapping based on the review schema.

    Args:
        review_schema (dict): The review schema containing the fields and their values.
        review_fields_mapping (dict): The current mapping of fields to the review data model.

    Returns:
        dict: The updated review fields mapping.

    """
    for field, content in review_schema.items():
        if field not in review_fields_mapping:
            not_correct_field = True
            while not_correct_field:
                input_text = f"Map the field {field} to the review data model\n"
                input_text += f"Some example values: {content['values'][:20]}"
                input_text += f"Possible fields are: {REVIEW_FIELDS}."
                review_data_model_field = input(f"{input_text} \n {field}:")
                if review_data_model_field in REVIEW_FIELDS:
                    not_correct_field = False
                else:
                    print(
                        f"Field {review_data_model_field} is not a valid review_field."
                    )

            if review_data_model_field:
                review_fields_mapping[field] = review_data_model_field
            else:
                review_fields_mapping[field] = None

    with open(os.path.join("./data", "review_fields_mapping.json"), "w+") as file:
        json.dump(review_fields_mapping, file, indent=4)

    return review_fields_mapping


def update_decision_fields_mapping(
    decision_schema: dict, decision_fields_mapping: dict
) -> dict:
    """
    Update the decision fields mapping based on the decision schema.

    Args:
        decision_schema (dict): The decision schema containing the field names and example values.
        decision_fields_mapping (dict): The current mapping of decision fields.

    Returns:
        dict: The updated decision fields mapping.
    """
    for decision_field, content in decision_schema.items():
        if decision_field not in decision_fields_mapping:
            not_correct_field = True
            while not_correct_field:
                decision_type = input(
                    f"Example values {content['values']}. Map the field  {decision_field} to the decision data model ({DECISION_FIELDS}):"
                )
                decision_fields_mapping[decision_field] = decision_type
                if decision_type in DECISION_FIELDS:
                    not_correct_field = False
                else:
                    print(f"Field {decision_type} is not a valid decision_field.")

    with open(os.path.join("./data", "decision_fields_mapping.json"), "w") as file:
        json.dump(decision_fields_mapping, file, indent=4)

    return decision_fields_mapping


def schemas2data_models(
    schemas: dict,
    venue_id: str,
    review_fields_mapping: dict,
    decision_fields_mapping: dict,
) -> None:
    """
    Builds encodings for OpenReview data based on the provided directory.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        None
    """

    directory = "./data"

    # Update venue2review_encoding
    if os.path.exists(os.path.join(directory, "venue2review_encodings.json")):
        with open(os.path.join(directory, "venue2review_encodings.json"), "r+") as file:
            venue2review_encodings = json.load(file)
    else:
        venue2review_encodings = {}

    if venue_id not in venue2review_encodings:
        all_encodings = {}
        for review_field, content in schemas["review_schema"].items():
            field_type = review_fields_mapping[review_field]
            if field_type in [
                "score",
                "confidence",
                "novelty",
                "correctness",
                "clarity",
                "impact",
                "reproducibility",
            ]:
                encoding = encoder(set(content["values"]))
                if len(encoding) == 0:
                    logging.warning(f"No encoding for {review_field}")
                    continue
                all_encodings[review_field] = encoding
        venue2review_encodings[venue_id] = all_encodings

        # Normalized scores of venue2review_encodings to be between 0 and 1
        all_normalized_encodings = {}
        for field, encoding in venue2review_encodings[venue_id].items():
            n_values = len(encoding)
            normalized_values = np.linspace(0, 1, n_values)
            value2rank = {
                value[0]: rank
                for rank, value in enumerate(
                    sorted(encoding.items(), key=lambda x: x[1])
                )
            }

            normalized_encoding = {}
            for key in encoding.keys():
                rank = value2rank[key]
                value = normalized_values[rank]
                normalized_encoding[key] = value
            encoding = normalized_encoding
            all_normalized_encodings[field] = encoding

        venue2review_encodings[venue_id] = all_normalized_encodings

        # Save updated venue2review_encodings
        with open(os.path.join(directory, "venue2review_encodings.json"), "w") as file:
            json.dump(venue2review_encodings, file, indent=4)

    # Update decision encodings
    if os.path.exists(os.path.join(directory, "venue2decision_encodings.json")):
        with open(
            os.path.join(directory, "venue2decision_encodings.json"), "r+"
        ) as file:
            venue2decision_encodings = json.load(file)
    else:
        venue2decision_encodings = {}

    if venue_id not in venue2decision_encodings:
        all_encodings = {}
        for decision_field, content in schemas["decision_schema"].items():
            field_type = decision_fields_mapping[decision_field]
            if field_type == "decision":
                encoding = {}
                input_text = (
                    f"Example values: {set(content['values'])} Skip this field (y/n):"
                )
                response = input(input_text)
                if response == "y":
                    continue

                for value in list(set(content["values"])):
                    encoding[value] = int(input(f"key: {value} (Reject=0/Accept=1):"))
                all_encodings[decision_field] = encoding

        venue2decision_encodings[venue_id] = encoding

        with open(
            os.path.join(directory, "venue2decision_encodings.json"), "w"
        ) as file:
            json.dump(venue2decision_encodings, file, indent=4)


def encoder(values: set) -> dict:
    """
    Encodes a list of values using a string split approach.

    Args:
        values (list): A list of values to be encoded.

    Returns:
        dict: A dictionary containing the encoded values.

    Raises:
        None

    Examples:
        >>> values = ["10:apple", "20:banana", "30:orange"]
        >>> string_split_encoder(values)
        {'10:apple': 10, '20:banana': 20, '30:orange': 30}
    """

    print("All values of the field:", values)
    encoding = {}
    for value in values:
        try:
            score = int(value)
        except Exception as e:
            try:
                score = int(value.split(":")[0])
            except Exception as e:
                try:
                    score = int(value.split(" ")[0])
                except Exception as e:
                    response = input(
                        "Skip this field, since it is not a numerical field (y/n):"
                    )
                    if response == "y":
                        return {}
                    else:
                        score = int(input(f"{value}:"))
        encoding[value] = score
    return encoding
