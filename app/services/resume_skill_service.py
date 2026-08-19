"""Resume-upload skill + competency enrichment.

When an employee's skill/competency record is thin (or an RM just wants to
enrich it), this lets someone upload that employee's real resume (PDF or
Word) and uses this app's own LLM provider chain (app/ai/llm.py -- same
Azure OpenAI/Gemini/Claude failover Buddy uses) to extract real skills AND
real competency evidence from it -- never scraped from LinkedIn/Google (a
real ToS and privacy problem for a tool meant to run on real employees),
only from a document the user actually provides.

Deliberately does NOT touch HR feedback (11_HR_Feedback_dummy.csv). That
table is structurally third-party: a specific real reviewer_employee_id
(a manager/EM/PM) giving their own opinion about one specific real
project_id. A resume is self-authored -- it can never genuinely contain
"what someone else said about this person on project X". Extracting
"feedback" from a resume would mean inventing a review, a rating, and a
reviewer that never existed, which is a materially different (and worse)
kind of fabrication than a resume-derived skill or competency. Skills and
competency are properties a resume can legitimately speak to; feedback is
not.

Every row added this way is tagged with a "resume_extracted" source --
scoring.py's existing observed-vs-not discount already treats anything
other than "observed" as lower-confidence, so a resume-derived record is
automatically weighted below a real project-verified one, consistent with
every other inferred source in this app.
"""
import io
import json
import shutil
import uuid
from datetime import datetime

import pandas as pd

from app.ai import llm
from app.core import db
from app.core.adapter import LocalAdapter
from app.core.config import APP_STATE_DIR, TRANSFORMED_DIR

RESUME_UPLOADS_DIR = APP_STATE_DIR / "resumes"
RESUME_UPLOADS_DIR.mkdir(exist_ok=True)

RESUME_BACKUPS_DIR = APP_STATE_DIR / "upload_backups"
RESUME_BACKUPS_DIR.mkdir(exist_ok=True)

# One JSON record per successful upload (resume or LinkedIn-exported PDF) --
# lets the UI show "you already imported this employee's document, here's
# what it added" instead of re-extracting blind on every click, and lets the
# raw file be re-served for an in-browser preview. Keyed by employee_id.
IMPORT_INDEX_PATH = APP_STATE_DIR / "document_imports.json"

SKILLS_CSV_PATH = TRANSFORMED_DIR / "05_Skill_Details_clean.csv"
COMPETENCY_CSV_PATH = TRANSFORMED_DIR / "06_Competency_Details_clean.csv"

# Keeps the LLM prompt bounded -- a real resume is a handful of pages at most;
# this is a safety cap, not a real-world limit anyone should hit.
MAX_RESUME_CHARS = 15000

# A skill someone lists on their own resume reflects real hands-on
# experience, not just passing familiarity -- 4/5 before the skill_source
# discount below (not 5/5, since it's still self-reported and unverified).
RESUME_SKILL_SCORE = 4

_CANONICAL_COES = ("Data Engineering", "AI & ML", "Full Stack Engineering", "TechOps & Automation", "BI & Reporting")

CONTENT_TYPES = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


class ResumeProcessingError(Exception):
    pass


def _load_import_index() -> dict:
    if not IMPORT_INDEX_PATH.exists():
        return {}
    try:
        return json.loads(IMPORT_INDEX_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_import_index(index: dict) -> None:
    IMPORT_INDEX_PATH.write_text(json.dumps(index, indent=2))


def list_document_imports(employee_id: str) -> list[dict]:
    """Every past successful resume/LinkedIn-PDF import for this employee,
    most recent first -- backs the "you already imported this" panel."""
    index = _load_import_index()
    records = index.get(employee_id, [])
    return sorted(records, key=lambda r: r["uploaded_at"], reverse=True)


def get_document_import_file(employee_id: str, import_id: str) -> tuple[bytes, str, str]:
    """Raw bytes + filename + content-type for one past import, for an
    in-browser preview (PDF) or download (Word)."""
    index = _load_import_index()
    records = index.get(employee_id, [])
    record = next((r for r in records if r["import_id"] == import_id), None)
    if record is None:
        raise ResumeProcessingError("No such import on record for this employee.")
    path = RESUME_UPLOADS_DIR / employee_id / record["stored_filename"]
    if not path.exists():
        raise ResumeProcessingError("The original file is no longer available on disk.")
    ext = record["filename"].rsplit(".", 1)[-1].lower() if "." in record["filename"] else ""
    return path.read_bytes(), record["filename"], CONTENT_TYPES.get(ext, "application/octet-stream")


def _extract_text(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise ResumeProcessingError(f"Could not read this PDF: {exc}") from exc
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == "docx":
        import docx
        try:
            doc = docx.Document(io.BytesIO(content))
        except Exception as exc:
            raise ResumeProcessingError(f"Could not read this Word document: {exc}") from exc
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ResumeProcessingError(f"Unsupported file type '.{ext}' -- upload a PDF or Word (.docx) resume.")

    text = text.strip()
    if not text:
        raise ResumeProcessingError(
            "Could not extract any real text from this file -- it may be a scanned image without a real text layer."
        )
    return text[:MAX_RESUME_CHARS]


def _real_competency_questions() -> list[str]:
    """The org's own real, curated behavioral-competency statements (excludes
    the "Tenure-based capability proxy" synthetic-proxy row, which isn't a
    real evaluated dimension) -- read live from the real competency table so
    this always reflects whatever the org's actual framework currently is,
    never a hardcoded copy that could drift from it."""
    df = pd.read_csv(COMPETENCY_CSV_PATH)
    real = df[df["competency_source"] == "observed"]
    return sorted(real["competency_question"].dropna().unique().tolist())


def _build_extraction_prompt(resume_text: str, competency_questions: list[str]) -> str:
    numbered_questions = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(competency_questions))
    return f"""You are extracting real, evidence-based information from a real employee's resume for an internal staffing tool.

PART 1 -- SKILLS
Extract ONLY skills, tools, technologies, platforms, and professional competencies that are ACTUALLY mentioned in the
resume text -- never invent or infer a skill that isn't genuinely present. For each, if it clearly belongs to one of
these Centers of Excellence, tag it; otherwise use null: Data Engineering, AI & ML, Full Stack Engineering, TechOps & Automation, BI & Reporting

PART 2 -- COMPETENCY EVIDENCE
Below is the org's real, fixed list of behavioral competency statements. For EACH one, decide if the resume text
provides genuine, specific evidence supporting it (e.g. a described project, responsibility, or outcome that clearly
demonstrates it) -- not a vague guess. Only include a competency in your answer if the evidence is real and specific;
if the resume says nothing that speaks to a statement, leave it out entirely (do not force-fit every statement).
Score 1-5 based on how strong and specific the evidence is (1 = weak/indirect mention, 5 = strong, explicit, repeated evidence).

Competency statements (use the EXACT text below, do not paraphrase):
{numbered_questions}

Respond with ONLY a JSON object, no other text, no markdown fences:
{{
  "skills": [{{"skill": "<exact skill name, title case>", "coe": "<one of the 5 CoE names above, or null>"}}],
  "competencies": [{{"question": "<EXACT statement text from the list above>", "score": <1-5>, "evidence": "<one short phrase from the resume that supports this>"}}],
  "summary": "<1-2 sentence real professional summary based only on the resume text>"
}}

Resume text:
---
{resume_text}
---
"""


def _call_llm_for_extraction(resume_text: str, competency_questions: list[str]) -> dict:
    providers = llm.get_providers()
    if not providers:
        raise ResumeProcessingError(
            "No AI provider is configured -- set GEMINI_API_KEY, ANTHROPIC_API_KEY, or the Azure OpenAI vars in the backend .env file."
        )

    prompt = _build_extraction_prompt(resume_text, competency_questions)
    messages = [{"role": "user", "content": prompt}]

    for provider in providers:
        try:
            turn = provider.generate_with_tools(messages, [], max_tokens=2000)
        except Exception:
            continue
        content = (turn or {}).get("content")
        if not content:
            continue
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
        try:
            parsed = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed.get("skills"), list):
            return parsed

    raise ResumeProcessingError("The AI could not extract a valid result from this resume -- try a different file.")


def _backup(path):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, RESUME_BACKUPS_DIR / f"{path.stem}.{stamp}{path.suffix}")


def _append_skills(employee_id: str, emp: pd.Series, raw_skills: list[dict], source_label: str) -> list[str]:
    existing = pd.read_csv(SKILLS_CSV_PATH)
    existing_lower = set(
        existing.loc[existing["employee_id"] == employee_id, "Skill"].astype(str).str.strip().str.lower()
    )

    new_rows = []
    added = []
    for s in raw_skills:
        skill_name = str(s["skill"]).strip()
        if not skill_name or skill_name.lower() in existing_lower:
            continue
        coe = s.get("coe") if s.get("coe") in _CANONICAL_COES else None
        new_rows.append({
            "employee_id": employee_id,
            "Designation": emp.get("job_name"),
            "COE": coe or emp.get("department_name"),
            "COE Skill": coe or "Resume",
            "Skill": skill_name,
            "SubSkill": skill_name,
            "Experience": None,
            "Score": RESUME_SKILL_SCORE,
            "skill_source": source_label,
        })
        added.append(skill_name)
        existing_lower.add(skill_name.lower())

    if new_rows:
        _backup(SKILLS_CSV_PATH)
        pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True).to_csv(SKILLS_CSV_PATH, index=False)
    return added


def _append_competencies(employee_id: str, emp: pd.Series, raw_competencies: list[dict], real_questions: set[str], source_label: str) -> list[str]:
    existing = pd.read_csv(COMPETENCY_CSV_PATH)
    existing_questions = set(
        existing.loc[existing["employee_id"] == employee_id, "competency_question"].astype(str).str.strip()
    )

    new_rows = []
    added = []
    for c in raw_competencies:
        question = str(c.get("question") or "").strip()
        # Only ever a question that's genuinely in the org's real fixed list
        # -- guards against the model paraphrasing or inventing a new one.
        if not question or question not in real_questions or question in existing_questions:
            continue
        try:
            score = max(1, min(5, int(round(float(c.get("score", 0))))))
        except (TypeError, ValueError):
            continue
        new_rows.append({
            "employee_id": employee_id,
            "designation": emp.get("job_name"),
            "coe_dep": emp.get("department_name"),
            "competency_sheet": emp.get("job_name"),
            "competency_question": question,
            "response": "Yes",
            "score": score,
            "competency_source": source_label,
        })
        added.append(question)
        existing_questions.add(question)

    if new_rows:
        _backup(COMPETENCY_CSV_PATH)
        pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True).to_csv(COMPETENCY_CSV_PATH, index=False)
    return added


def process_resume(employee_id: str, filename: str, content: bytes, channel: str = "resume") -> dict:
    channel = channel if channel in ("resume", "linkedin") else "resume"
    source_label = f"{channel}_extracted"

    employees = LocalAdapter().get_employees()
    match = employees[employees["employee_id"] == employee_id]
    if match.empty:
        raise ResumeProcessingError(f"No real employee found with id '{employee_id}'.")
    emp = match.iloc[0]

    resume_text = _extract_text(filename, content)
    competency_questions = _real_competency_questions()
    result = _call_llm_for_extraction(resume_text, competency_questions)

    raw_skills = [s for s in result.get("skills", []) if isinstance(s, dict) and str(s.get("skill") or "").strip()]
    raw_competencies = [c for c in result.get("competencies", []) if isinstance(c, dict)]
    if not raw_skills and not raw_competencies:
        raise ResumeProcessingError("No real skills or competency evidence could be extracted from this resume.")

    import_id = uuid.uuid4().hex
    emp_dir = RESUME_UPLOADS_DIR / employee_id
    emp_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{import_id}_{filename}"
    (emp_dir / stored_filename).write_bytes(content)

    added_skills = _append_skills(employee_id, emp, raw_skills, source_label)
    added_competencies = _append_competencies(employee_id, emp, raw_competencies, set(competency_questions), source_label)

    if added_skills or added_competencies:
        db.reload()

    record = {
        "import_id": import_id,
        "employee_id": employee_id,
        "channel": channel,
        "filename": filename,
        "stored_filename": stored_filename,
        "uploaded_at": datetime.now().isoformat(),
        "summary": result.get("summary"),
        "extracted_skill_count": len(raw_skills),
        "added_skills": added_skills,
        "skipped_existing_skill_count": len(raw_skills) - len(added_skills),
        "extracted_competency_count": len(raw_competencies),
        "added_competencies": added_competencies,
        "skipped_existing_competency_count": len(raw_competencies) - len(added_competencies),
    }
    index = _load_import_index()
    index.setdefault(employee_id, []).append(record)
    _save_import_index(index)

    return record
