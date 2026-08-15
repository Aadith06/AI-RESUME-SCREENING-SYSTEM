"""
app.py

Recruiter tool: paste a job description, upload many resumes at once,
and get a ranked shortlist of the best-fit candidates based on semantic
similarity to the JD and overlap with required skills.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import pandas as pd
import streamlit as st

from utils import ResumeParser, ResumeMatcher


def render_html(html: str):
    """
    st.markdown(..., unsafe_allow_html=True) runs the string through a
    Markdown parser first. Markdown treats any line indented 4+ spaces as
    a literal code block, which breaks multi-line HTML templates written
    with normal Python indentation. Stripping leading whitespace from
    every line avoids that entirely.
    """
    flat = "\n".join(line.strip() for line in html.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------- #
# Page setup + design system
# ---------------------------------------------------------------------- #
st.set_page_config(page_title="Shortlist — Resume Ranker", page_icon="🗂️", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink: #EDEFF2;
    --ink-soft: #93A1B0;
    --paper: #0E1620;
    --panel: #182430;
    --card: #141E28;
    --line: #26333F;
    --signal: #3FBFA8;
    --signal-soft: rgba(63, 191, 168, 0.12);
    --gold: #E0B04D;
    --gold-soft: rgba(224, 176, 77, 0.10);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--paper);
}

/* Force Streamlit's own text elements to the light ink color on dark bg */
p, span, label, .stMarkdown, .stCaption, div[data-testid="stMarkdownContainer"] {
    color: var(--ink);
}

h1, h2, h3, .brand-title {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.01em;
}

/* ---- Header band ---- */
.header-band {
    background: var(--panel);
    margin: -1rem -1rem 2rem -1rem;
    padding: 2.2rem 2.6rem 1.8rem 2.6rem;
    border-bottom: 3px solid var(--signal);
}
.brand-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    color: var(--signal);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.brand-title {
    color: var(--ink);
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.brand-sub {
    color: var(--ink-soft);
    font-size: 0.95rem;
    margin-top: 0.35rem;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ---- Section panels ---- */
.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--signal);
    margin-bottom: 0.4rem;
    font-weight: 600;
}

/* ---- Leaderboard cards ---- */
.candidate-card {
    display: flex;
    gap: 1.1rem;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.85rem;
    align-items: flex-start;
}
.candidate-card.top {
    border: 1px solid var(--gold);
    background: linear-gradient(180deg, var(--gold-soft) 0%, var(--card) 60%);
}
.rank-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--ink-soft);
    min-width: 2.2rem;
    line-height: 1.1;
}
.candidate-card.top .rank-num {
    color: var(--gold);
}
.candidate-body {
    flex: 1;
}
.candidate-name-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.candidate-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.08rem;
    color: var(--ink);
}
.top-ribbon {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: var(--gold);
    color: #1A1305;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
}
.candidate-meta {
    font-size: 0.82rem;
    color: var(--ink-soft);
    margin-top: 0.15rem;
}
.match-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-top: 0.7rem;
}
.match-track {
    flex: 1;
    height: 8px;
    background: var(--panel);
    border-radius: 6px;
    overflow: hidden;
}
.match-fill {
    height: 100%;
    background: var(--signal);
    border-radius: 6px;
}
.match-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--signal);
    min-width: 3rem;
    text-align: right;
}
.skill-tags {
    margin-top: 0.6rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}
.tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 0.18rem 0.55rem;
    border-radius: 20px;
    background: var(--signal-soft);
    color: var(--signal);
    border: 1px solid rgba(63, 191, 168, 0.35);
}
.tag.miss {
    background: rgba(255, 255, 255, 0.03);
    color: #5C6B7A;
    border: 1px solid var(--line);
    text-decoration: line-through;
}
.tag.extra {
    background: rgba(224, 176, 77, 0.10);
    color: var(--gold);
    border: 1px solid rgba(224, 176, 77, 0.35);
}
.summary-line {
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--ink);
    margin-top: 0.75rem;
    padding: 0.6rem 0.8rem;
    background: var(--panel);
    border-radius: 8px;
    border-left: 3px solid var(--signal);
}
.required-label, .extra-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-top: 0.7rem;
    margin-bottom: 0.35rem;
}

/* ---- Empty state ---- */
.empty-state {
    border: 1px dashed var(--line);
    border-radius: 10px;
    padding: 2.2rem;
    text-align: center;
    color: var(--ink-soft);
    background: var(--card);
}

/* Streamlit widget tweaks */
.stButton>button {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    background: var(--signal);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
}
.stButton>button:hover {
    background: #2E9683;
    color: white;
}

/* Text input / textarea / uploader on dark surfaces */
.stTextArea textarea, .stFileUploader, div[data-testid="stFileUploaderDropzone"] {
    background: var(--card) !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_html(
    """
    <div class="header-band">
        <div class="brand-eyebrow">RESUME INTELLIGENCE</div>
        <p class="brand-title">Shortlist</p>
        <div class="brand-sub">Paste a job description, upload a stack of resumes, get back a ranked shortlist.</div>
    </div>
    """
)


# ---------------------------------------------------------------------- #
# Cached resources
# ---------------------------------------------------------------------- #
@st.cache_resource
def get_parser():
    return ResumeParser()


@st.cache_resource
def get_matcher():
    return ResumeMatcher()


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def parse_uploaded_resumes(uploaded_files, parser) -> tuple[list[dict], list[str]]:
    parsed, errors = [], []
    progress = st.progress(0.0, text="Reading resumes...")
    for i, uploaded_file in enumerate(uploaded_files):
        tmp_path = save_uploaded_file(uploaded_file)
        try:
            result = parser.parse_resume(tmp_path)
            result["filename"] = uploaded_file.name
            if result["raw_text"]:
                parsed.append(result)
            else:
                errors.append(f"{uploaded_file.name}: no extractable text.")
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")
        finally:
            os.remove(tmp_path)
        progress.progress((i + 1) / len(uploaded_files), text=f"Read {i + 1}/{len(uploaded_files)}")
    progress.empty()
    return parsed, errors


def skill_tag_html(required_skills: list[str], matched_skills: list[str]) -> str:
    matched_set = set(matched_skills)
    tags = []
    for skill in required_skills:
        cls = "tag" if skill.lower() in matched_set else "tag miss"
        tags.append(f'<span class="{cls}">{skill}</span>')
    return "".join(tags) if tags else '<span class="candidate-meta">No required skills detected in JD.</span>'


def extra_skill_tag_html(required_skills: list[str], candidate_skills: list[str]) -> str:
    """Skills the candidate has that weren't asked for in the JD — a sign of extra breadth."""
    required_set = {s.lower() for s in required_skills}
    extras = [s for s in candidate_skills if s.lower() not in required_set]
    if not extras:
        return ""
    tags = "".join(f'<span class="tag extra">{s}</span>' for s in extras)
    return f'<div class="extra-label">Also brings</div><div class="skill-tags">{tags}</div>'


def fit_tier(pct: int) -> str:
    if pct >= 80:
        return "Excellent fit"
    if pct >= 60:
        return "Strong fit"
    if pct >= 40:
        return "Moderate fit"
    return "Weak fit"


def candidate_summary(r: dict, required_skills: list[str], pct: int) -> str:
    """A one-line, plain-English readout a recruiter can skim without decoding tags."""
    name = r["name"] or r["filename"]
    matched = r["matched_skills"]
    required_set = {s.lower() for s in required_skills}
    missing = [s for s in required_skills if s.lower() not in {m.lower() for m in matched}]
    tier = fit_tier(pct)

    if not required_skills:
        return f"{tier} ({pct}%). No specific required skills were detected in the job description to compare against."

    if not matched:
        return (
            f"{tier} ({pct}%). {name.split()[0] if r['name'] else 'This candidate'} does not clearly "
            f"demonstrate any of the {len(required_skills)} required skills for this role."
        )

    matched_preview = ", ".join(matched[:4])
    sentence = (
        f"{tier} ({pct}%). Covers {len(matched)} of {len(required_skills)} required skills, "
        f"including {matched_preview}."
    )
    if missing:
        missing_preview = ", ".join(missing[:3])
        sentence += f" Missing: {missing_preview}{'...' if len(missing) > 3 else ''}."
    return sentence


def render_candidate_card(rank: int, r: dict, required_skills: list[str]):
    is_top = rank == 1
    pct = round(r["final_score"] * 100)
    ribbon = '<span class="top-ribbon">Top pick</span>' if is_top else ""
    summary = candidate_summary(r, required_skills, pct)
    extras_html = extra_skill_tag_html(required_skills, r["skills"])

    render_html(
        f"""
        <div class="candidate-card {'top' if is_top else ''}">
            <div class="rank-num">{rank:02d}</div>
            <div class="candidate-body">
                <div class="candidate-name-row">
                    <div class="candidate-name">{r['name'] or r['filename']}</div>
                    {ribbon}
                </div>
                <div class="candidate-meta">
                    {r['email'] or 'No email found'} &nbsp;·&nbsp; {r['phone'] or 'No phone found'} &nbsp;·&nbsp; {r['filename']}
                </div>
                <div class="match-row">
                    <div class="match-track"><div class="match-fill" style="width:{pct}%;"></div></div>
                    <div class="match-pct">{pct}%</div>
                </div>
                <div class="summary-line">{summary}</div>
                <div class="required-label">Required skills</div>
                <div class="skill-tags">{skill_tag_html(required_skills, r['matched_skills'])}</div>
                {extras_html}
            </div>
        </div>
        """
    )


def render_results(ranked: list[dict], required_skills: list[str]):
    render_html('<div class="panel-label">Shortlist — ranked by fit</div>')

    for rank, r in enumerate(ranked, start=1):
        render_candidate_card(rank, r, required_skills)

    table_rows = [{
        "Rank": i,
        "Name": r["name"] or "Unknown",
        "File": r["filename"],
        "Email": r["email"] or "-",
        "Phone": r["phone"] or "-",
        "Final Score": r["final_score"],
        "Semantic Score": r["semantic_score"],
        "Skill Score": r["skill_score"],
        "Matched Skills": ", ".join(r["matched_skills"]) or "-",
    } for i, r in enumerate(ranked, start=1)]

    df = pd.DataFrame(table_rows)
    st.download_button(
        "Download shortlist as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="shortlist.csv",
        mime="text/csv",
    )

    with st.expander("View full extracted skills per candidate"):
        for r in ranked:
            st.markdown(f"**{r['name'] or r['filename']}** — {', '.join(r['skills']) or 'None found'}")


# ---------------------------------------------------------------------- #
# Intake panels
# ---------------------------------------------------------------------- #
col_jd, col_upload = st.columns([1.1, 1], gap="large")

with col_jd:
    render_html('<div class="panel-label">01 · Job description</div>')
    job_description = st.text_area(
        "Job description",
        height=260,
        placeholder="Paste the full job description here — responsibilities, required skills, everything.",
        label_visibility="collapsed",
    )

with col_upload:
    render_html('<div class="panel-label">02 · Candidate resumes</div>')
    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) ready.")

st.write("")
rank_clicked = st.button("Rank candidates", type="primary")
st.write("")


# ---------------------------------------------------------------------- #
# Run + results
# ---------------------------------------------------------------------- #
if not rank_clicked:
    render_html(
        """
        <div class="empty-state">
            No shortlist yet. Paste a job description and upload resumes above, then click <b>Rank candidates</b>.
        </div>
        """
    )
else:
    if not job_description.strip():
        st.error("Please paste a job description first.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        parser = get_parser()
        parsed_resumes, errors = parse_uploaded_resumes(uploaded_files, parser)

        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) had issues"):
                for e in errors:
                    st.write(f"- {e}")

        if not parsed_resumes:
            st.error("No resumes could be parsed successfully.")
        else:
            with st.spinner("Scoring candidates against the job description..."):
                matcher = get_matcher()
                ranked = matcher.rank_candidates(job_description, parsed_resumes)
                required_skills = ranked[0]["required_skills"] if ranked else []

            render_results(ranked, required_skills)
