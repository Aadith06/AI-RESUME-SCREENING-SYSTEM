"""
skills.py

Extracts predefined technical skills from text using spaCy's
token-level PhraseMatcher (fast, case-insensitive phrase matching).
"""

import spacy
from spacy.matcher import PhraseMatcher

DEFAULT_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "ruby", "golang",
    "sql", "nosql", "mongodb", "postgresql", "mysql",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy",
    "aws", "gcp", "azure", "docker", "kubernetes", "git", "ci/cd",
    "power bi", "tableau", "excel", "agile", "scrum", "linux", "unix",
    "data analysis", "data visualization", "artificial intelligence",
]


class SkillExtractor:
    """Extracts a known set of skills from free-text using PhraseMatcher."""

    def __init__(self, custom_skills: list[str] | None = None, spacy_model: str = "en_core_web_sm"):
        try:
            # Disable unneeded pipeline components for a speed boost
            self.nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        except OSError as exc:
            raise OSError(f"spaCy model '{spacy_model}' not found.") from exc

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.skills_db = [s.lower() for s in (custom_skills or DEFAULT_SKILLS)]

        skill_patterns = list(self.nlp.pipe(self.skills_db))
        self.matcher.add("TECH_SKILLS", skill_patterns)

    def extract_skills(self, text: str) -> list[str]:
        """Scans input text and returns a unique, sorted list of matched skills."""
        if not text or not isinstance(text, str):
            return []

        doc = self.nlp(text)
        matches = self.matcher(doc)

        extracted_skills = {doc[start:end].text.title() for _, start, end in matches}
        return sorted(extracted_skills)


if __name__ == "__main__":
    extractor = SkillExtractor()
    sample_text = "Proficient in pandas, numpy, SQL, and deep learning."
    print("\n--- EXTRACTED SKILLS ---")
    for skill in extractor.extract_skills(sample_text):
        print(f"- {skill}")
