"""The pipeline that builds the OpenReview dataset."""

import argparse
import logging
import os

import yaml

from openreview_parser.scientific_databases.openreview import (
    get_openreview_client,
)
from openreview_parser.pipeline.submissions import submissions2papers
from openreview_parser.pipeline.venue import get_venue_instances
from openreview_parser.pipeline.schema import get_schemas
from openreview_parser.utils.data import VenueInstance


DEBUG_VENUE = VenueInstance(
    venue="ICLR.cc/2024/Conference",
    name="ICLR.cc",
    year=2024,
    conference=True,
    workshop=False,
    workshop_name=None,
)


def build_dataset(config: dict) -> None:
    """
    Build the dataset by performing various steps including data retrieval, preprocessing, and transformation.

    Args:.
        config (dict): The configuration dictionary containing various settings.

    Returns:
        None
    """
    # Get OpenReview client
    openreview_client = get_openreview_client(config)

    # Prepare Directory Structure for Dataset
    for directory in ["venues", "papers", "schemas", "pdfs", "xmls"]:
        os.makedirs(os.path.join(config["save_directory"], directory), exist_ok=True)

    # Get all venue instances
    venue_instances = get_venue_instances(openreview_client, config)
    logging.info(f"Obtained venue instances")

    if config["venue"] == "":
        venue_instances = {vi for vi in venue_instances if vi.venue == config["venue"]}

    if config["debug"]:
        venue_instances = {DEBUG_VENUE}

    # Get schema for venue instances
    venue_instances_full_info = get_schemas(openreview_client, venue_instances, config)
    logging.info(f"Obtained schemas")

    if config["venue"] == "":
        venue_instances_full_info = {
            vi for vi in venue_instances_full_info if vi.venue == config["venue"]
        }

    if config["debug"]:
        venue_instances_full_info = {DEBUG_VENUE}

    # Parse submissions into data model for each venue instance
    submissions2papers(venue_instances_full_info, openreview_client, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenReview dataset pipeline")
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    parser.add_argument("--venue", type=str, help="Parse only this venue", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if args.venue is not None:
        assert config["debug"] == False, "Cannot specify venue when in debug"
        config["venue"] = args.venue

    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler("logs/complete_dataset.log")],
    )

    build_dataset(config)
