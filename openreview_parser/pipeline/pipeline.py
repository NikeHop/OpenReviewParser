"""
The pipeline that builds the OpenReview dataset
"""

import argparse
import logging
import os

import yaml
import tqdm

from openreview_parser.scientific_databases.openreview_v2 import (
    get_openreview_client_v2,
)
from openreview_parser.pipeline.submissions import submissions2papers
from openreview_parser.pipeline.venue import get_venue_instances
from openreview_parser.pipeline.schema import get_schemas
from openreview_parser.utils.data import VenueInstance

DEBUG_VENUE = VenueInstance(
    venue="NeurIPS.cc/2023/Conference",
    name="NeurIPS.cc",
    year=2023,
    conference=True,
    workshop=False,
    workshop_name=None,
)

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
    Builds the dataset by performing various steps including data retrieval, preprocessing, and transformation.

    Args:
        client (openreview.Client): The OpenReview client object.
        config (dict): The configuration dictionary containing various settings.

    Returns:
        None
    """

    # Get OpenReview client
    openreview_client = get_openreview_client_v2(config["openreview"])

    # Prepare Directory Structure for Dataset
    for directory in ["venues", "dataset", "papers", "schemas", "pdfs", "xmls"]:
        os.makedirs(os.path.join(config["save_directory"], directory), exist_ok=True)

    # Get all venue instances
    venue_instances = get_venue_instances(openreview_client, config)
    logging.info(f"Obtained venue instances")

    if config["debug"]:
        venue_instances = [DEBUG_VENUE]

    # Get schema for venue instances
    venue_instances_full_info = get_schemas(openreview_client, venue_instances, config)
    logging.info(f"Obtained schemas")
    print(venue_instances_full_info)
    print([type(venue) for venue in venue_instances_full_info])

    # Parse submissions into data model for each venue instance
    submissions2papers(venue_instances_full_info, openreview_client, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenReview dataset pipeline")
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler("logs/complete_dataset.log")],
    )

    build_dataset(config)
