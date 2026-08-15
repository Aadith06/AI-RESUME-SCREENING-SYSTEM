"""
parser.py

Extracts raw text from resume files (PDF / DOCX) and parses out
structured information: name, email, phone, and document sections.
"""

import os
import re
import logging
import spacy
import pdfplumber

logger = logging.getLogger(__name__)

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# Regex patterns compiled once at module load (faster, cleaner)
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')


class ResumeParser:
    """
    Extracts raw text from resume files and parses out structured
    information such as contact details, name, and document sections.
    """

    # Ordered longest-heading-first so multi-word headings (e.g. "WORK
    # EXPERIENCE") are matched before their shorter substrings ("EXPERIENCE").
    SECTION_HEADINGS = sorted(
        [
            "EDUCATION", "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT",
            "PROJECTS", "CERTIFICATIONS", "SKILLS", "TECHNICAL SKILLS",
            "SUMMARY", "OBJECTIVE", "PUBLICATIONS", "LANGUAGES",
        ],
        key=len,
        reverse=True,
    )

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Loads the spaCy NER model used for name extraction."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{spacy_model}' not found. "
                f"Run: python -m spacy download {spacy_model}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Text extraction
    # ------------------------------------------------------------------ #
    def extract_text(self, file_path: str) -> str:
        """Determines file type and extracts raw text accordingly."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        ext = os.path.splitext(file_path)[-1].lower().lstrip(".")

        if ext == "pdf":
            return self._extract_from_pdf(file_path)
        if ext == "docx":
            if not DOCX_SUPPORT:
                raise ImportError(
                    "python-docx is not installed. Run: pip install python-docx"
                )
            return self._extract_from_docx(file_path)

        raise ValueError(
            f"Unsupported file format '.{ext}'. Supported formats: .pdf, .docx"
        )

    def _extract_from_pdf(self, file_path: str) -> str:
        text_chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_chunks.append(extracted)
        except Exception as exc:
            raise RuntimeError(f"Failed to read PDF '{file_path}': {exc}") from exc
        return "\n".join(text_chunks).strip()

    def _extract_from_docx(self, file_path: str) -> str:
        try:
            document = docx.Document(file_path)
            return "\n".join(p.text for p in document.paragraphs).strip()
        except Exception as exc:
            raise RuntimeError(f"Failed to read DOCX '{file_path}': {exc}") from exc

    # ------------------------------------------------------------------ #
    # Field extraction
    # ------------------------------------------------------------------ #
    def extract_email(self, text: str) -> str | None:
        """Returns the first matched email address, or None."""
        match = EMAIL_PATTERN.search(text)
        return match.group(0) if match else None

    def extract_phone(self, text: str) -> str | None:
        """Returns the first matched phone number, or None."""
        match = PHONE_PATTERN.search(text)
        return match.group(0).strip() if match else None

    def extract_name(self, text: str) -> str | None:
        """Extracts the candidate's name via spaCy NER from the top of the resume."""
        doc = self.nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip().split()) > 1:
                return ent.text.strip()
        return None

    def extract_sections(self, text: str) -> dict:
        """Splits text into logical sections based on known heading keywords."""
        sections: dict[str, list[str]] = {"UNKNOWN": []}
        current_section = "UNKNOWN"

        for line in text.split("\n"):
            clean_line = line.strip().upper()

            matched_heading = next(
                (
                    heading for heading in self.SECTION_HEADINGS
                    if heading in clean_line and len(clean_line) < 30
                ),
                None,
            )

            if matched_heading:
                current_section = matched_heading
                sections.setdefault(current_section, [])
            elif line.strip():
                sections[current_section].append(line.strip())

        return {section: " ".join(lines) for section, lines in sections.items()}

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def parse_resume(self, file_path: str) -> dict:
        """Runs the full parsing pipeline on a single resume file."""
        raw_text = self.extract_text(file_path)

        if not raw_text:
            logger.warning("No text could be extracted from %s", file_path)

        return {
            "name": self.extract_name(raw_text),
            "email": self.extract_email(raw_text),
            "phone": self.extract_phone(raw_text),
            "sections": self.extract_sections(raw_text),
            "raw_text": raw_text,
        }


if __name__ == "__main__":
    print("[INFO] Parser module ready to be imported.")
