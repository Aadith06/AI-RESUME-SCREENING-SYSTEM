"""
exploration.py

Exploratory Data Analysis (EDA) for the resume dataset (columns:
ID, Resume_str, Resume_html, Category).
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class ResumeDatasetAnalyzer:
    """Handles Exploratory Data Analysis (EDA) of the resume dataset."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """Loads the CSV file into a DataFrame, with an encoding fallback."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The dataset file at {self.file_path} was not found.")

        try:
            self.df = pd.read_csv(self.file_path, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed, retrying with latin-1 encoding.")
            self.df = pd.read_csv(self.file_path, encoding="latin-1")
        except Exception as exc:
            raise IOError(f"Error reading the CSV file: {exc}") from exc

        print(f"[INFO] Dataset successfully loaded. Shape: {self.df.shape}")
        return self.df

    def _require_data(self):
        if self.df is None:
            raise RuntimeError("Data not loaded yet. Call load_data() first.")

    def get_basic_info(self):
        """Prints structural data types, null values, and basic statistics."""
        self._require_data()

        print("\n" + "=" * 50)
        print("1. DATASET STRUCTURE & MISSING VALUES")
        print("=" * 50)
        print(self.df.info())

        print("\nMissing Values per Column:")
        print(self.df.isnull().sum())

    def analyze_categories(self, top_n: int = 10):
        """Analyzes the distribution of target job categories."""
        self._require_data()

        if "Category" not in self.df.columns:
            print("[ERROR] 'Category' column missing from dataset.")
            return

        print("\n" + "=" * 50)
        print(f"2. TOP {top_n} CATEGORY DISTRIBUTION")
        print("=" * 50)
        counts = self.df["Category"].value_counts()
        percentages = self.df["Category"].value_counts(normalize=True) * 100

        distribution_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages.round(2)})
        print(distribution_df.head(top_n))
        print(f"\nTotal Unique Categories: {self.df['Category'].nunique()}")

    def analyze_text_length(self, target_col: str = "Resume_str"):
        """Computes text length statistics (character and word counts)."""
        self._require_data()

        if target_col not in self.df.columns:
            print(f"[WARNING] '{target_col}' not found.")
            return

        print("\n" + "=" * 50)
        print("3. TEXT LENGTH & COMPLEXITY ANALYSIS")
        print("=" * 50)

        clean_text_series = self.df[target_col].dropna().astype(str)
        char_counts = clean_text_series.apply(len)
        word_counts = clean_text_series.apply(lambda x: len(x.split()))

        metrics_df = pd.DataFrame({
            "Metric": [
                "Average Char Count", "Max Char Count", "Min Char Count",
                "Average Word Count", "Max Word Count", "Min Word Count",
            ],
            "Value": [
                int(char_counts.mean()), char_counts.max(), char_counts.min(),
                int(word_counts.mean()), word_counts.max(), word_counts.min(),
            ],
        })
        print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    # Portable path: looks for the CSV in the same 'dataset' folder as this
    # script, so the project runs on any machine without editing this path.
    DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resume.csv")

    try:
        analyzer = ResumeDatasetAnalyzer(file_path=DATASET_PATH)
        analyzer.load_data()
        analyzer.get_basic_info()
        analyzer.analyze_categories(top_n=10)
        analyzer.analyze_text_length()
    except Exception as error:
        print(f"[FATAL ERROR] Pipeline failed: {error}")
