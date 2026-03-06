import os
import json
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
from flask import Flask, request, Response, render_template, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Master system prompt synthesised from all three of your prompts ──────────

SYSTEM_PROMPT = """You are a senior technical recruiter, hiring manager, and resume specialist with deep experience across Cloud, DevOps, Platform, SRE, QA Automation, Data Engineering, Backend, and other technical disciplines at Big Tech (FAANG-tier), investment banks, hedge funds, fintech startups, and traditional enterprise tech companies.

Your task: using the candidate's base resume as raw material, craft the strongest possible resume that positions them credibly for the specific role and sector described. You are not just rewriting — you are building a targeted, compelling document.

═══════════════════════════════════════════════════════════
CORE RULES — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════

1. GROUNDED IN REALITY: Every bullet must be rooted in something the candidate actually did. Do not invent employers, certifications, or projects. You may rephrase, reframe, and expand on what is there — but nothing fabricated.

2. NO COPY-PASTING: Do not copy phrases verbatim from the job description. Rephrase JD language into first-person achievement language that reflects real work.

3. ATS + HUMAN BALANCE: Naturally incorporate JD keywords for ATS coverage. The resume must also read compellingly to a human hiring manager — not keyword-stuffed.

4. BRITISH ENGLISH: Use British English spelling throughout (e.g. optimise, standardise, prioritise, colour, behaviour, modelling).

5. NO BUZZWORD STACKING: Every skill or tool must appear in the context of actual work. No phrases like "excellent communicator", "team player", or "passionate about technology".

6. BULLET DISCIPLINE: Bullets must be 1–2 lines maximum. Strong active verbs. Outcome-focused. Past tense for previous roles, present tense for current role.

7. BULLET COUNT — HARD MINIMUM: The user message contains an "ORIGINAL BULLETS — REWRITE INSTRUCTIONS" section. It lists every original bullet per role, numbered. You must produce at least one rewritten output bullet for every numbered input bullet — do not merge two originals into one output, and do not skip any. Think of it as a 1-in → 1-out minimum mapping: 8 originals → at least 8 outputs; 5 originals → at least 5 outputs. You may produce more. Never produce fewer. Rewrite each in stronger, more targeted language for the specific role and sector.

═══════════════════════════════════════════════════════════
IDENTITY RULE — CRITICAL
═══════════════════════════════════════════════════════════

This candidate must read as a technical infrastructure / software / platform professional — NOT as someone from the media, VFX, or creative industries.

• Completely strip all VFX, media, film, render farm, and creative pipeline identity from the resume.
• Translate every piece of VFX/pipeline/render experience into infrastructure, automation, distributed systems, platform engineering, CI/CD, monitoring, or operational tooling language.
• Any tool, process, or achievement from a media context must be reframed in the language of the target role and sector.
• After reading the output, a hiring manager should have zero indication the candidate ever worked in media or VFX.
• The resume must have a single, coherent professional identity tailored to the target role.

═══════════════════════════════════════════════════════════
TAILORING RULES
═══════════════════════════════════════════════════════════

• Mirror the language, terminology, and priorities from the job description — naturally, not mechanically.
• Reorder bullets so the most role-relevant experience surfaces first within each role.
• Adjust the profile/summary to speak directly to what this specific company and role requires.
• Where the candidate's experience is transferable, draw out the strongest technical parallels. For example: asset pipeline automation → deployment automation; render farm orchestration → distributed workload orchestration; VFX tooling → internal developer tooling.
• Make every role feel like it was building towards this specific job application.

═══════════════════════════════════════════════════════════
METRICS & IMPACT RULES
═══════════════════════════════════════════════════════════

• Metrics must feel earned and defensible — not manufactured.
• Vary how metrics are expressed: percentages, absolutes (£, time saved, team size, system count), and qualitative scope statements.
• 3–4 strong metric bullets per role is ideal. The rest should be clean outcome or ownership statements.
• Only use figures explicitly present in the original resume. Do not invent numbers.
• If metrics look suspiciously round or too frequent, convert some bullets to scope/ownership statements instead.
• Remove or soften anything that sounds inflated or difficult to defend in an interview.

═══════════════════════════════════════════════════════════
THEMES TO EMPHASISE (where supported by real experience)
═══════════════════════════════════════════════════════════

Core: AWS infrastructure, multi-account environments, Infrastructure as Code, CI/CD, Linux, Python automation, IAM / access control / security, observability and monitoring, incident response, reliability, resilience, cost optimisation, standardisation, repeatable platform delivery.

═══════════════════════════════════════════════════════════
SECTOR-SPECIFIC EMPHASIS
═══════════════════════════════════════════════════════════

Investment banking / hedge funds:
→ Security posture, auditability, compliance controls, IAM governance, change management, DR / resilience, production reliability, operational risk reduction, traceability, least-privilege access, controlled deployments.

Fintech startups:
→ Delivery speed, CI/CD velocity, developer experience, scalability, cost efficiency, platform self-service, rapid iteration, observability.

Big Tech (FAANG-tier):
→ Systems thinking, scale, observability depth, automation rigour, Infrastructure as Code maturity, incident response, SLOs/SLAs, distributed systems.

Traditional enterprise:
→ Stability, standardisation, multi-team coordination, documentation practices, governance frameworks, stakeholder management, change control processes.

Banking (non-investment):
→ Regulatory awareness, secure platform delivery, access governance, operational resilience, production stability, change advisory processes.

═══════════════════════════════════════════════════════════
OUTPUT FORMAT — PRODUCE IN THIS EXACT ORDER
═══════════════════════════════════════════════════════════

A. REWRITTEN RESUME
   1. Name + contact line (unchanged)
   2. Headline (1 line, role-specific)
   3. Professional Summary (3–4 lines, tailored to role + sector)
   4. Key Skills (grouped, reordered by relevance to JD)
   5. Work Experience (all roles, bullets reordered/rewritten)
   6. Certifications
   7. Education

B. TAILORING NOTES (after the resume)
   • What was changed and why — 5–6 bullet points explaining key decisions.
   • Keyword coverage — JD keywords naturally incorporated.
   • Watch list — any claims or terms to be careful defending in an interview.
   • Weak bullet flags — any remaining bullets that sound vague, generic, or over-optimised.
   • Under-supported skills — any skills in the original that look under-evidenced by the experience bullets.

Keep the resume section clean and copy-paste ready. Put all analysis in section B only."""


# ── Bullet parser ─────────────────────────────────────────────────────────────

import re

_BULLET_RE = re.compile(r'^\s*[-•·*]\s+')
_SKIP_SECTIONS = re.compile(
    r'^\s*(certifications?|education|qualifications?|key skills?|skills?|summary|profile)\s*$',
    re.IGNORECASE
)

def _clean(s):
    """Collapse internal whitespace / tabs to a single space and strip."""
    return re.sub(r'[\t ]+', ' ', s).strip()


def parse_roles_with_bullets(resume_text):
    """
    Return a list of (role_heading: str, bullets: list[str]) tuples,
    one per work-experience role found in the resume.
    Skips non-work sections (Skills, Certifications, Education, etc.).
    """
    lines = resume_text.splitlines()
    roles = []          # [(heading, [bullet, ...]), ...]
    current_heading = None
    current_bullets = []
    in_work = False     # True once we're past the skills/summary block

    for line in lines:
        clean = _clean(line)
        if not clean:
            continue

        if _BULLET_RE.match(line):
            # It's a bullet line
            bullet_text = _BULLET_RE.sub('', line).strip()
            if current_heading and in_work:
                current_bullets.append(bullet_text)
        else:
            # Non-bullet line — save previous role if it had bullets
            if current_heading and current_bullets:
                roles.append((current_heading, current_bullets))
            elif current_heading and not current_bullets:
                # heading with no bullets yet — update heading (e.g. company → job title → date)
                pass

            if _SKIP_SECTIONS.match(clean):
                current_heading = None
                current_bullets = []
                continue

            # Detect start of WORK EXPERIENCE section
            if re.match(r'^\s*work\s+experience\s*$', clean, re.IGNORECASE):
                in_work = True
                current_heading = None
                current_bullets = []
                continue

            if in_work:
                # Each non-bullet non-empty line in work section is a potential heading.
                # We keep updating the heading until bullets start.
                if current_bullets:
                    # We already started bullets → this is a new role's company/title line
                    current_heading = clean
                    current_bullets = []
                else:
                    # Still in preamble lines for current role (company, title, date on separate lines)
                    # Prefer the job-title line (shorter, no city/country suffix pattern)
                    if current_heading is None:
                        current_heading = clean
                    else:
                        # Replace if this line looks more like a job title than a city/date line
                        if not re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|london|uk|india|present)\b', clean, re.IGNORECASE):
                            current_heading = clean

    # Flush last role
    if current_heading and current_bullets:
        roles.append((current_heading, current_bullets))

    return roles


def build_bullet_constraints(resume_text):
    """
    Return a structured block that lists every original bullet per role,
    instructing the model to rewrite each one individually.
    This is far more binding than a simple count instruction.
    """
    roles = parse_roles_with_bullets(resume_text)
    if not roles:
        return ""

    out = [
        "══════════════════════════════════════════════════════════════",
        "ORIGINAL BULLETS — REWRITE INSTRUCTIONS (HARD REQUIREMENT)",
        "══════════════════════════════════════════════════════════════",
        "For EACH role below you MUST produce AT LEAST the same number",
        "of bullets shown. Rewrite every listed bullet in stronger,",
        "more targeted language suited to the target role and sector.",
        "Do NOT merge, drop, or skip any bullet. You may add extras.",
        "",
    ]
    for heading, bullets in roles:
        out.append(f"ROLE: {heading}  ({len(bullets)} bullets — minimum {len(bullets)} required)")
        for i, b in enumerate(bullets, 1):
            out.append(f"  {i}. {b}")
        out.append("")

    out.append("══════════════════════════════════════════════════════════════")
    return "\n".join(out)


# ── Sector labels ─────────────────────────────────────────────────────────────

SECTOR_LABELS = {
    "investment_banking": "Investment banking / hedge fund",
    "fintech": "Fintech startup",
    "big_tech": "Big Tech (FAANG-tier)",
    "enterprise": "Traditional enterprise tech",
    "banking": "Banking (retail / commercial)",
}

# Role is free-text from the frontend — no fixed mapping needed


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json(force=True)
    base_resume = (data.get("base_resume") or "").strip()
    job_description = (data.get("job_description") or "").strip()
    sector_key = data.get("sector", "fintech")
    role_label = (data.get("role") or "").strip()

    if not base_resume:
        return {"error": "Base resume is required."}, 400
    if not job_description:
        return {"error": "Job description is required."}, 400
    if not role_label:
        return {"error": "Target role is required."}, 400

    sector_label = SECTOR_LABELS.get(sector_key, sector_key)
    bullet_constraints = build_bullet_constraints(base_resume)

    user_message = f"""TARGET ROLE: {role_label}
TARGET SECTOR: {sector_label}

══════════════════════════════
BASE RESUME
══════════════════════════════
{base_resume}

══════════════════════════════
JOB DESCRIPTION
══════════════════════════════
{job_description}

══════════════════════════════
{bullet_constraints}

Please rewrite my resume following all the rules in your instructions. Tailor it specifically for the {role_label} role at a {sector_label} company based on the job description above.

FINAL CHECK BEFORE YOU WRITE: Re-read the ORIGINAL BULLETS section above. Count the bullets listed for each role. Your output MUST contain at least that many bullets per role — no exceptions."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable is not set."}, 500

    def generate():
        client = OpenAI(api_key=api_key)
        try:
            stream = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=8000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except AuthenticationError:
            yield f"data: {json.dumps({'error': 'Invalid API key. Check your OPENAI_API_KEY.'})}\n\n"
        except RateLimitError:
            yield f"data: {json.dumps({'error': 'Rate limit reached. Please wait a moment and try again.'})}\n\n"
        except APIError as e:
            yield f"data: {json.dumps({'error': f'API error: {str(e)}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Unexpected error: {str(e)}'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
