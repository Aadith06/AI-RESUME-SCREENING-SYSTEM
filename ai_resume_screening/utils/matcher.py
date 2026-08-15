"""
matcher.py

Scores and ranks resumes against a job description using two signals:
  1. Semantic similarity (sentence-transformers embeddings + cosine similarity)
  2. Skill overlap (required skills mentioned in the JD vs skills found in the resume)

The final score is a weighted blend of both, so a resume that's a strong
overall fit AND covers the specific required skills ranks highest.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .skills import SkillExtractor


class ResumeMatcher:
    """Ranks a batch of resumes against a single job description."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        skill_extractor: SkillExtractor | None = None,
        semantic_weight: float = 0.6,
        skill_weight: float = 0.4,
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.semantic_weight = semantic_weight
        self.skill_weight = skill_weight

    def _semantic_scores(self, job_text: str, resume_texts: list[str]) -> list[float]:
        """Cosine similarity between the JD embedding and each resume embedding, scaled to 0-1."""
        if not resume_texts:
            return []

        embeddings = self.embedder.encode([job_text] + resume_texts)
        job_embedding = embeddings[0:1]
        resume_embeddings = embeddings[1:]

        sims = cosine_similarity(job_embedding, resume_embeddings)[0]
        # Cosine similarity is already roughly in [0, 1] for these embeddings,
        # but clip defensively in case of small negative values.
        return [max(0.0, min(1.0, float(s))) for s in sims]

    def _skill_score(self, required_skills: list[str], candidate_skills: list[str]) -> float:
        """Fraction of required skills the candidate's resume actually contains."""
        if not required_skills:
            return 0.0
        required_set = {s.lower() for s in required_skills}
        candidate_set = {s.lower() for s in candidate_skills}
        matched = required_set & candidate_set
        return len(matched) / len(required_set)

    def rank_candidates(self, job_description: str, resumes: list[dict]) -> list[dict]:
        """
        Args:
            job_description: Raw text of the job description.
            resumes: List of dicts, each with at least:
                {"filename": str, "raw_text": str, "name": str|None,
                 "email": str|None, "phone": str|None}

        Returns:
            The same list of dicts, enriched with "skills", "matched_skills",
            "semantic_score", "skill_score", and "final_score", sorted by
            final_score descending (best candidate first).
        """
        required_skills = self.skill_extractor.extract_skills(job_description)
        resume_texts = [r["raw_text"] for r in resumes]

        semantic_scores = self._semantic_scores(job_description, resume_texts)

        enriched = []
        for resume, semantic_score in zip(resumes, semantic_scores):
            candidate_skills = self.skill_extractor.extract_skills(resume["raw_text"])
            skill_score = self._skill_score(required_skills, candidate_skills)
            matched_skills = sorted(
                {s.lower() for s in required_skills} & {s.lower() for s in candidate_skills}
            )

            final_score = (
                self.semantic_weight * semantic_score + self.skill_weight * skill_score
            )

            enriched.append({
                **resume,
                "skills": candidate_skills,
                "matched_skills": matched_skills,
                "semantic_score": round(semantic_score, 4),
                "skill_score": round(skill_score, 4),
                "final_score": round(final_score, 4),
                "required_skills": required_skills,
            })

        enriched.sort(key=lambda r: r["final_score"], reverse=True)
        return enriched
