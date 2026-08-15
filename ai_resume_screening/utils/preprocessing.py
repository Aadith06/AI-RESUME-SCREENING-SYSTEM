"""
preprocessing.py

A text-cleaning pipeline for resumes and job descriptions: lowercasing,
noise removal, stopword filtering, and lemmatization.
"""

import re
import logging
import spacy
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
EMAIL_PATTERN = re.compile(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b')
# Strip punctuation but keep '+' and '#' so skills like "C++" / "C#" survive
NON_ALPHANUM_PATTERN = re.compile(r'[^a-z0-9\s\+#]')


class TextPreprocessor:
    """Cleans and lemmatizes resume / job-description text."""

    def __init__(self, spacy_model: str = "en_core_web_sm", max_length: int = 1_500_000):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError as exc:
            raise OSError(
                f"spaCy model '{spacy_model}' not found. "
                f"Run: python -m spacy download {spacy_model}"
            ) from exc

        # Set once here, not on every clean_text() call
        self.nlp.max_length = max_length

        self._ensure_stopwords()
        self.stop_words = set(stopwords.words("english"))

    @staticmethod
    def _ensure_stopwords():
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords", quiet=True)

    def clean_text(self, text: str) -> str:
        """Runs the full cleaning pipeline on a given string."""
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = URL_PATTERN.sub("", text)
        text = EMAIL_PATTERN.sub("", text)
        text = NON_ALPHANUM_PATTERN.sub(" ", text)

        doc = self.nlp(text)
        cleaned_tokens = [
            token.lemma_
            for token in doc
            if token.text not in self.stop_words and not token.is_space
        ]

        return " ".join(cleaned_tokens)

    def clean_batch(self, texts: list[str], batch_size: int = 50, n_process: int = 1) -> list[str]:
        """
        Cleans many documents efficiently using spaCy's nlp.pipe instead of
        calling clean_text() in a Python loop (much faster for large datasets).
        """
        prepped = []
        for text in texts:
            if not text or not isinstance(text, str):
                prepped.append("")
                continue
            t = text.lower()
            t = URL_PATTERN.sub("", t)
            t = EMAIL_PATTERN.sub("", t)
            t = NON_ALPHANUM_PATTERN.sub(" ", t)
            prepped.append(t)

        results = []
        for doc in self.nlp.pipe(prepped, batch_size=batch_size, n_process=n_process):
            tokens = [
                token.lemma_
                for token in doc
                if token.text not in self.stop_words and not token.is_space
            ]
            results.append(" ".join(tokens))
        return results


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample_resume_text = "Experienced in C++, Python, and Machine Learning! Visit my site at https://github.com"
    print("--- RAW TEXT ---")
    print(sample_resume_text)
    print("\n--- CLEANED TEXT ---")
    print(preprocessor.clean_text(sample_resume_text))
