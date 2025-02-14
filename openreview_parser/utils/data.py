"""PyDantic Data Models for Unification of OpenReview data."""

from collections import defaultdict
from typing import Any

from fuzzywuzzy import process
from pydantic import BaseModel
from pydantic import validator


class VenueInstance(BaseModel):
    """Represents an instance of a venue (e.g., a conference or workshop)."""

    venue: str
    name: str
    year: int
    conference: bool
    workshop: bool
    workshop_name: str | None = None

    def __hash__(self) -> int:
        """
        Calculate the hash value of the object based on its venue.

        Returns:
            int: The hash value of the object.
        """
        return hash(self.venue)

    def __eq__(self, other: Any) -> bool:
        """
        Check if the current VenueInstance is equal to the given object.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.

        Raises:
            NotImplementedError: If the comparison is not implemented for the type of the given object.
        """
        if isinstance(other, VenueInstance):
            return self.venue == other.venue
        elif isinstance(other, str):
            return self.venue == other
        else:
            raise NotImplementedError(
                f"Comparison for object of type {type(other)} not implemented"
            )


class Affiliation(BaseModel):
    """Represents an affiliation of a person."""

    laboratory: str | dict | None = None
    institution: str | dict | None = None
    location: str | dict | None = None


class TextReview(BaseModel):
    """
    Represents a text review of a paper.

    Attributes:
        title (str, optional): The title of the review.
        paper_summary (str, optional): The summary of the paper being reviewed.
        main_review (str, optional): The main body of the review.
        strength_weakness (str, optional): The strengths and weaknesses of the paper.
        questions (str, optional): Any questions raised by the reviewer.
        limitations (str, optional): The limitations of the review.
        review_summary (str, optional): A summary of the review.
    """

    title: str | None = None
    paper_summary: str | None = None
    main_review: str | None = None
    strength_weakness: str | None = None
    questions: str | None = None
    limitations: str | None = None
    review_summary: str | None = None


class Review(BaseModel):
    """
    Represents a review object with various attributes.

    Attributes:
        review_id (str): The OpenReview ID of the review.
        review (TextReview): The text content of the review.
        score (float, optional): The score assigned to the review. Must be between 0 and 1.
        confidence (float, optional): The confidence level of the review. Must be between 0 and 1.
        novelty (float, optional): The novelty rating of the review. Must be between 0 and 1.
        correctness (float, optional): The correctness rating of the review. Must be between 0 and 1.
        clarity (float, optional): The clarity rating of the review. Must be between 0 and 1.
        impact (float, optional): The impact rating of the review. Must be between 0 and 1.
        reproducibility (float, optional): The reproducibility rating of the review. Must be between 0 and 1.
        ethics (str, optional): The ethics assessment of the review.
    """

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
        """
        Check if the given score is valid.

        Parameters:
            v (float): The score to be checked.

        Returns:
            float: The validated score.

        Raises:
            ValueError: If the score is not between 0 and 1.
        """
        if v == None:
            return v
        if v < 0 or v > 1:
            raise ValueError("score must be between 0 and 1")
        return v


class Comment(BaseModel):
    """
    Represents a comment object.

    Attributes:
        title (str, optional): The title of the comment. Defaults to None.
        comment (str): The content of the comment.
    """

    title: str | None = None
    comment: str


class Reference(BaseModel):
    """
    Represents a reference to a paper.

    Attributes:
        title (str): The title of the paper.
        abstract (str, optional): The abstract of the paper. Defaults to an empty string.
        authors (list[str]): The authors of the paper.
        arxiv_id (str, optional): The arXiv ID of the paper. Defaults to an empty string.
        s2_corpus_id (str, optional): The Semantic Scholar corpus ID of the paper. Defaults to an empty string.
        external_ids (dict, optional): Additional external IDs associated with the paper. Defaults to an empty dictionary.
        intents (list[str], optional): The intents associated with the reference. Defaults to None.
        isInfluential (bool, optional): Indicates whether the reference is influential. Defaults to None.
    """

    title: str
    abstract: str = ""
    authors: list[str]
    arxiv_id: str | None = ""
    s2_corpus_id: str | None = ""
    external_ids: dict | None = {}
    intents: list[str] | None = None
    isInfluential: bool | None = None


class Section(BaseModel):
    """
    Represents a section of a document.

    Attributes:
        name (str): The name of the section.
        sec_num (str): The section number.
        classification (str, optional): The classification of the section.
        text (str): The text content of the section.
        subsections (list[Section], optional): The list of subsections contained within the section.
    """

    name: str
    sec_num: str
    classification: str = ""
    text: str
    subsections: list["Section"] = []


class Paper(BaseModel):
    """
    Represents a paper with various attributes and methods for organizing and retrieving information.

    Attributes:
        title (str): The title of the paper.
        authors (list[str]): The list of authors of the paper.
        abstract (str, optional): The abstract of the paper. Defaults to None.
        summary (str, optional): The summary of the paper. Defaults to None.
        paperhash (str): The hash value of the paper.
        arxiv_id (str, optional): The arXiv ID of the paper. Defaults to None.
        s2_corpus_id (str, optional): The Semantic Scholar corpus ID of the paper. Defaults to "".
        field_of_study (list[str] or str, optional): The field of study of the paper. Defaults to None.
        venue (str, optional): The venue where the paper was published. Defaults to None.
        publication_date (str, optional): The publication date of the paper. Defaults to None.
        n_references (int, optional): The number of references in the paper. Defaults to None.
        n_citations (int, optional): The number of citations of the paper. Defaults to None.
        n_influential_citations (int, optional): The number of influential citations of the paper. Defaults to None.
        external_ids (dict, optional): The external IDs associated with the paper. Defaults to None.
        parsed_pdf (dict, optional): The parsed PDF content of the paper. Defaults to None.
        structured_content (dict[str, Section]): The structured content of the paper. Defaults to an empty dictionary.
        decision (bool, optional): The decision of the paper. Defaults to None.
        decision_text (str, optional): The decision text of the paper. Defaults to None.
        reviews (list[Review], optional): The list of reviews of the paper. Defaults to None.
        comments (list[Comment], optional): The list of comments on the paper. Defaults to None.
        references (list[Reference], optional): The list of references of the paper. Defaults to None.
        section_name2section (dict): A mapping of section names to their corresponding sections.
        bibref2section (dict): A mapping of bibliography references to their corresponding sections.
        bibref2paperhash (dict): A mapping of bibliography references to their corresponding paper hashes.
        hypothesis (str, optional): The hypothesis of the paper. Defaults to None.
        license (str, optional): The license of the paper. Defaults to None.
    """

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

    # License
    license: str | None = None

    def organize_text(self) -> None:
        """Organize the parsed_pdf into a dictionary."""
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
        Return the text representation of the structured content, including the main paper and optional appendix.

        Args:
            with_appendix (bool): Whether to include the appendix in the returned text. Default is True.

        Returns:
            str: The text representation of the structured content.
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
        """
        Return a list of section names from the structured content.

        Returns:
            list[str]: A list of section names.
        """
        section_names = []
        for section in self.structured_content.values():
            section_names.append(section.name)
        return section_names

    def get_section_by_name(self, section_name: str) -> str:
        """
        Retrieve the text content of a section with the given name.

        Args:
            section_name (str): The name of the section to retrieve.

        Returns:
            str: The text content of the section, or an empty string if the section is not found.
        """
        for section in self.structured_content.values():
            if section.name == section_name:
                return section.text
        return ""

    def get_section_by_classification(self, section_type: str) -> str:
        """
        Retrieve the text of a section based on its classification.

        Args:
            section_type (str): The classification of the section to retrieve.

        Returns:
            str: The text of the section with the specified classification, or an empty string if not found.
        """
        for section in self.structured_content.values():
            if section.classification == section_type:
                return section.text
        return ""

    def create_section_name2section(self):
        """
        Create a dictionary mapping section names to their corresponding classifications.

        This method iterates over the structured content and populates the `section_name2section` dictionary
        with section names and their corresponding classifications. It also includes subsection names in the
        dictionary.

        Returns:
            None
        """
        self.section_name2section = {}
        for section in self.structured_content.values():
            section_name = section.name.lower()
            sec_cls = section.classification
            self.section_name2section[section_name] = sec_cls

            for subsection in section.subsections:
                self.section_name2section[subsection.name] = sec_cls

    def create_bibref2section(self):
        """
        Create a mapping of reference IDs to sections in the parsed PDF.

        This method populates the `bibref2section` dictionary, which maps reference IDs to sections in the parsed PDF.
        It iterates over the elements in the parsed PDF's body text and determines the section to which each element belongs.
        If a section name is found in the `section_name2section` dictionary, the element is associated with that section.
        The method also counts the number of occurrences of each reference ID in each section.

        Note: This method assumes that the `section_name2section` dictionary and the parsed PDF are already populated.

        Returns:
            None
        """
        self.bibref2section = {}
        self.create_section_name2section()

        if not (
            len(self.section_name2section) == 0
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
        """
        Create a mapping between bibliographic references and paper hashes.

        This method populates the `bibref2paperhash` dictionary with key-value pairs,
        where the key is a bibliographic reference and the value is the corresponding
        paper hash. The paper hash is generated based on the title and authors of the paper.

        Note: This method requires the `parsed_pdf` attribute to be properly initialized.

        Returns:
            None
        """
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
    """
    Parse the author dictionary and returns the formatted name as a string.

    Args:
        author (dict): A dictionary containing the author's information.

    Returns:
        str: The formatted name of the author.

    """
    name = (
        author["first"].lower()
        + " "
        + " ".join(author["middle"]).lower()
        + " "
        + author["last"].lower()
    )
    return name


def get_paperhash(title: str, authors: list[str]) -> str:
    """
    Generate a unique hash for a paper based on its title and authors.

    Args:
        title (str): The title of the paper.
        authors (list[str]): The list of authors of the paper.

    Returns:
        str: The generated paper hash.

    """
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
