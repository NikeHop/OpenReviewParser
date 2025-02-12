"""
PyDantic Data Models for Unification of OpenReview Data
"""

import logging

from collections import defaultdict
from typing import Any

from fuzzywuzzy import process
from pydantic import BaseModel
from pydantic import validator


class VenueInstance(BaseModel):
    venue: str
    name: str
    year: int
    conference: bool
    workshop: bool
    workshop_name: str | None = None

    def __hash__(self) -> int:
        return hash(self.venue)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, VenueInstance):
            return self.venue == other.venue
        elif isinstance(other, str):
            return self.venue == other
        else:
            raise NotImplementedError(
                f"Comparison for object of type {type(other)} not implemented"
            )


class Affiliation(BaseModel):
    laboratory: str | dict | None = None
    institution: str | dict | None = None
    location: str | dict | None = None


class TextReview(BaseModel):
    title: str | None = None
    paper_summary: str | None = None
    main_review: str | None = None
    strength_weakness: str | None = None
    questions: str | None = None
    limitations: str | None = None
    review_summary: str | None = None


class Review(BaseModel):
    review_id: str
    review: TextReview
    score: float | None = None
    confidence: float | None = None
    novelty: float | None = None
    correctness: float | None = None
    clarity: float | None = None
    impact: float | None = None
    reproducibility: float | None = None
    ethics: str | None = None

    @validator(
        "score",
        "confidence",
        "novelty",
        "correctness",
        "clarity",
        "impact",
        "reproducibility",
    )
    @classmethod
    def check_score(cls, v: float) -> float:
        if v == None:
            return v
        if v < 0 or v > 1:
            raise ValueError("score must be between 0 and 1")
        return v


class Comment(BaseModel):
    title: str | None = None
    comment: str


class Reference(BaseModel):
    # Basic paper info
    title: str
    abstract: str = ""
    authors: list[str]

    # IDs
    arxiv_id: str | None = ""
    s2_corpus_id: str | None = ""
    external_ids: dict | None = {}

    # Reference specific info
    intents: list[str] | None = None
    isInfluential: bool | None = None


class Section(BaseModel):
    name: str
    sec_num: str
    classification: str = ""
    text: str
    subsections: list["Section"] = []


class Paper(BaseModel):
    # Basic paper info
    title: str
    authors: list[str]
    abstract: str | None = None
    summary: str | None = None

    # ID's
    paperhash: str
    arxiv_id: str | None = None
    s2_corpus_id: str | None = ""

    # OpenReview Metadata
    field_of_study: list[str] | str | None = None
    venue: str | None = None
    publication_date: str | None = None

    # s2 Metadata
    n_references: int | None = None
    n_citations: int | None = None
    n_influential_citations: int | None = None
    external_ids: dict | None = None

    # Content
    parsed_pdf: dict | None = None
    structured_content: dict[str, Section] = {}

    # Review Data
    decision: bool | None = None
    decision_text: str | None = None
    reviews: list[Review] | None = None
    comments: list[Comment] | None = None

    # References
    references: list[Reference] | None = None
    section_name2section: dict = {}
    bibref2section: dict = {}
    bibref2paperhash: dict = {}

    # Hypothesis
    hypothesis: str | None = None

    def organize_text(self) -> None:
        """
        Organizes the parsed_pdf into a dictionary.
        """

        conclusion_passed = False
        last_sec_num = "Unnumbered"
        if (
            self.parsed_pdf is None
            or "pdf_parse" not in self.parsed_pdf
            or self.parsed_pdf["pdf_parse"] is None
            or "body_text" not in self.parsed_pdf["pdf_parse"]
        ):
            self.structured_content = {}
            return None

        for part in self.parsed_pdf["pdf_parse"]["body_text"]:
            # Update conclusion passed
            if not conclusion_passed and len(self.structured_content) > 0:
                sec_names = [sec.name for sec in self.structured_content.values()]
                match, _ = fuzzy_matching("conclusion", sec_names)
                if match != "":
                    conclusion_passed = True

            sec_num = part.get("sec_num")
            section = part.get("section")
            section = section.lower() if section is not None else section

            if sec_num is None:
                sec_num = "appendix" if conclusion_passed else last_sec_num
            else:
                last_sec_num = sec_num

            main_sec, _, sub_sec = sec_num.partition(".")

            # Main sec is not present yet
            if main_sec not in self.structured_content:
                self.structured_content[main_sec] = Section(
                    name=section, sec_num=main_sec, text=part["text"]
                )
                if sub_sec != None:
                    self.structured_content[main_sec].subsections.append(
                        Section(name=section, sec_num=sub_sec, text=part["text"])
                    )
            else:
                # Add text to the section
                self.structured_content[main_sec].text += part["text"]
                if sub_sec != None:
                    # Add subsection
                    self.structured_content[main_sec].subsections.append(
                        Section(name=section, sec_num=sub_sec, text=part["text"])
                    )

    def get_text(self, with_appendix: bool = True) -> str:
        """
        Return the text of the paper.
        """
        text = ""

        paper_no_appendix = {
            key: value
            for key, value in self.structured_content.items()
            if value.sec_num != "appendix"
        }
        paper_appendix = {
            key: value
            for key, value in self.structured_content.items()
            if value.sec_num == "appendix"
        }

        for value in sorted(
            paper_no_appendix.values(),
            key=lambda y: int(y.sec_num) if y.sec_num != "Unnumbered" else float("inf"),
        ):
            text += f"{value.name}: \n {value.text} \n"

        appendix = ""
        for value in paper_appendix.values():
            appendix += value.text

        if with_appendix:
            text = f"Main paper: \n {text} \n Appendix: {appendix} \n"
        else:
            text = f"Main paper: \n {text} \n"

        return text

    def get_section_names(self) -> list[str]:
        section_names = []
        for section in self.structured_content.values():
            section_names.append(section.name)
        return section_names

    def get_section_by_name(self, section_name: str) -> str:
        for section in self.structured_content.values():
            if section.name == section_name:
                return section.text
        return ""

    def get_section_by_classification(self, section_type: str) -> str:
        for section in self.structured_content.values():
            if section.classification == section_type:
                return section.text
        return ""

    def create_section_name2section(self):
        self.section_name2section = {}
        for section in self.structured_content.values():
            section_name = section.name.lower()
            sec_cls = section.classification
            self.section_name2section[section_name] = sec_cls

            for subsection in section.subsections:
                self.section_name2section[subsection.name] = section

    def create_bibref2section(self):
        self.bibref2section = {}
        self.create_section_name2section()

        if not (
            len(self.section_name2section) != 0
            or self.parsed_pdf is None
            or "pdf_parse" not in self.parsed_pdf
            or self.parsed_pdf["pdf_parse"] is None
            or "body_text" not in self.parsed_pdf["pdf_parse"]
        ):
            self.bibref2section = defaultdict(lambda: defaultdict(int))
            for elem in self.parsed_pdf["pdf_parse"]["body_text"]:
                # Understand which section this element belongs to
                section_name = elem["section"].lower()
                if section_name in self.section_name2section:
                    section = self.section_name2section[section_name]
                else:
                    continue

                if "cite_spans" in elem:
                    for citation in elem["cite_spans"]:
                        if "ref_id" not in citation:
                            continue
                        ref_id = citation["ref_id"]
                        # Add everything to the mapping
                        self.bibref2section[ref_id][section] += 1

    def create_bibref2paperhash(self):
        self.bibref2paperhash = {}

        if not (
            self.parsed_pdf is None
            or "pdf_parse" not in self.parsed_pdf
            or self.parsed_pdf["pdf_parse"] is None
            or "bib_entries" not in self.parsed_pdf["pdf_parse"]
        ):
            for key, value in self.parsed_pdf["pdf_parse"]["bib_entries"].items():
                title = value["title"]
                authors = [
                    parse_author_bibentries(author) for author in value["authors"]
                ]
                paperhash = get_paperhash(title, authors)
                self.bibref2paperhash[key] = paperhash


def parse_author_bibentries(author: dict) -> str:
    name = (
        author["first"].lower()
        + " "
        + " ".join(author["middle"]).lower()
        + " "
        + author["last"].lower()
    )
    return name


def get_paperhash(title: str, authors: list[str]) -> str:
    if len(authors) == 0:
        return "|" + "_".join(title.split(" ")).lower()
    first_author_last_name = authors[0].split(" ")[-1]
    return first_author_last_name + "|" + "_".join(title.split(" ")).lower()


def fuzzy_matching(
    query: str, choices: list[str], threshold: int = 80
) -> tuple[str, int]:
    """
    Fuzzy matches a query against a list of choices.

    Args:
        query (str): The query to match.
        choices (list[str]): A list of choices to match against.

    Returns:
        str: The best match from the list of choices.
    """

    result = process.extractOne(query, choices)

    if result is None:
        return "", 0
    else:
        best_match, score = result

    if score < threshold:
        return "", score

    return best_match, score
