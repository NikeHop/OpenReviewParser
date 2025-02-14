"""
Extracting the venue instances from OpenReview and saving them to a CSV file.
"""
import json
import logging
import os

import openreview

from openreview.api import OpenReviewClient

from openreview_parser.utils.data import VenueInstance


def get_venue_instances(client: OpenReviewClient, config: dict) -> list[VenueInstance]:
    """
    Retrieves venue instances from OpenReview and saves them to a CSV file.

    Args:
        client (OpenReviewClient): The OpenReviewClient.
        config (dict): The relevant configuration.

    Returns:
        list[VenueInstance]: A list of VenueInstance objects.
    """

    # Load existing venues
    venue_instances, failed_venue_strings, venue_strings_api_v1 = load_venue_dataset(
        config
    )
    new_venue_instances = 0
    n_failed_venue_instances = 0  # Number of venue strings that could not be parsed

    # Get all venues from OpenReview
    venues = client.get_group(id="venues").members
    for venue in venues:
        if venue in venue_instances or venue in failed_venue_strings:
            continue

        if config["use_openreview_api_v2"]:
            if venue in venue_strings_api_v1:
                continue
        else:
            if venue not in venue_strings_api_v1:
                continue

        venue_elements = venue.split("/")

        # Extract the year
        year = get_year(venue_elements)
        if year == -1:
            logging.warning(f"Year could not be identified for venue {venue}")
            n_failed_venue_instances += 1
            failed_venue_strings.add(venue)
            continue

        # Extract venue type
        conference = is_conference_instance(venue_elements)
        workshop = is_workshop_instance(venue_elements)

        if not conference and not workshop:
            logging.warning(
                f"Venue {venue} could not be identified as a conference or workshop"
            )
            n_failed_venue_instances += 1
            failed_venue_strings.add(venue)
            continue

        # Extract venue name
        venue_name = get_venue_name(venue_elements, conference, workshop)
        if workshop:
            workshop_name = get_workshop_name(venue_elements)
        else:
            workshop_name = None

        vi = VenueInstance(
            venue=venue,
            name=venue_name,
            year=year,
            conference=conference,
            workshop=workshop,
            workshop_name=workshop_name,
        )

        venue_instances.add(vi)
        new_venue_instances += 1

    logging.info(f"Failed to parse {n_failed_venue_instances} venue strings")
    logging.info(f"Retrieved {new_venue_instances} new venue instances")

    # Save the dataset
    venue_instances_filepath = os.path.join(
        config["save_directory"], "venues", "venues.json"
    )
    with open(venue_instances_filepath, "w+") as file:
        json.dump([vi.model_dump() for vi in venue_instances], file, indent=4)

    failed_venue_strings_filepath = os.path.join(
        config["save_directory"], "venues", "failed_venue_strings.json"
    )
    with open(failed_venue_strings_filepath, "w+") as file:
        json.dump(list(failed_venue_strings), file, indent=4)

    return list(venue_instances)


def load_venue_dataset(config: dict) -> tuple[set[VenueInstance], set[str], set[str]]:
    """
    Load the existing venue data.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple[set,set,set]: A tuple, first element set of VenueInstance objects, second element set of failed venue strings, venus that can only be parsed with version 1 of the OpenReview API.
    """

    venue_instances = []
    venue_instances_filepath = os.path.join(
        config["save_directory"], "venues", "venues.json"
    )
    if os.path.exists(venue_instances_filepath):
        with open(venue_instances_filepath, "r") as file:
            venue_instances = json.load(file)
            venue_instances = [VenueInstance(**vi) for vi in venue_instances]

    failed_venue_strings = set()
    failed_venue_strings_filepath = os.path.join(
        config["save_directory"], "venues", "failed_venue_strings.json"
    )
    if os.path.exists(failed_venue_strings_filepath):
        with open(failed_venue_strings_filepath, "r") as file:
            failed_venue_strings = json.load(file)
            failed_venue_strings = set(failed_venue_strings)

    venue_strings_api_v1 = set()
    with open("data/venue_strings_api_v1.json", "r") as file:
        venue_strings_api_v1 = json.load(file)
        venue_strings_api_v1 = set(venue_strings_api_v1)

    return set(venue_instances), set(failed_venue_strings), set(venue_strings_api_v1)


def get_year(venue_elements: list[str]) -> int:
    """
    Extracts the year from the venue elements.

    Args:
        venue_elements (list): The elements of the venue.

    Returns:
        int: The year of the venue.
    """
    for element in venue_elements:
        if element.isdigit():
            return int(element)
    return -1


def is_workshop_instance(venue_elements: list[str]) -> bool:
    """
    Checks if the venue is a workshop instance.

    Args:
        venue_elements (list): The elements of the venue.

    Returns:
        bool: True if the venue is a workshop instance, False otherwise.
    """
    for element in venue_elements:
        if "workshop" == element.lower():
            return True
    return False


def is_conference_instance(venue_elements: list[str]) -> bool:
    """
    Checks if the venue is a conference instance.

    Args:
        venue_elements (list): The elements of the venue.

    Returns:
        bool: True if the venue is a conference instance, False otherwise.
    """
    for element in venue_elements:
        if "conference" == element.lower():
            return True
    return False


def get_workshop_name(venue_elements: list[str]) -> str:
    """
    Retrieves the name of the workshop.

    Args:
        venue_elements (list): The elements of the venue.

    Returns:
        str: The name of the workshop.
    """
    # Identify all elements before the Workshop element
    for i, element in enumerate(venue_elements):
        if "workshop" == element.lower():
            break_point = i
            break
    workshop_name = "-".join(venue_elements[break_point + 1 :])
    return workshop_name


def get_venue_name(venue_elements: list[str], conference: bool, workshop: bool) -> str:
    """
    Retrieves the name of the venue.

    Args:
        venue_elements (list): The elements of the venue.
        conference (bool): True if the venue is a conference instance, False otherwise.
        workshop (bool): True if the venue is a workshop instance, False otherwise.

    Returns:
        str: The name of the venue.
    """
    # Identify all elements before the Conference/Workshop element
    for i, element in enumerate(venue_elements):
        if conference and "conference" == element.lower():
            break_point = i
            break
        if workshop and "workshop" == element.lower():
            break_point = i
            break
    venue_elements = venue_elements[:break_point]

    # Remove the year from these elements
    venue_elements = [element for element in venue_elements if not element.isdigit()]

    # Join elements to form the venue name
    venue_name = "-".join(venue_elements)

    return venue_name
