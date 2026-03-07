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

6. BULLET DISCIPLINE: Strong active verbs. Outcome-focused. Past tense for previous roles, present tense for current role. Bullets should be 1–2 lines — but "1–2 lines" means CONTENT-RICH lines, not stripped-down summaries. For senior technical roles a 30–50 word bullet with specific tools, methods, and outcomes is correct. Do NOT condense a rich original bullet into a vague one-liner. Preserve all named technologies, tools, metrics, and technical specifics from the original. Depth and specificity make bullets credible; vagueness makes them weak.

7. BULLET COUNT — SLOT TEMPLATE: The user message includes a "WORK EXPERIENCE SKELETON" with ◆SLOT-N markers. Each ◆SLOT-N shows the full original bullet text as context. Replace each ◆SLOT-N with exactly one rewritten output bullet — stronger, more sector-targeted, but equally or more detailed than the original. Never merge two ◆SLOTs. Never delete a ◆SLOT line. 1-slot → 1-bullet minimum. You may add extra bullets after the last slot.

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
• PRESERVE TECHNICAL SPECIFICITY: Every named tool, service, framework, metric, and methodology from the original bullet must appear in the rewritten version (unless explicitly replaced by a stronger equivalent). A rewrite that drops "CloudFormation", "MCP server", "EKS", or "£30k/year" is strictly worse than the original. Rewrites must add clarity and sector framing, never strip detail.

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


def build_we_skeleton(resume_text):
    """
    Build a pre-structured Work Experience skeleton with one ◆SLOT per
    original bullet.  The model fills in every slot — it cannot silently
    skip one without leaving a visible ◆SLOT marker in the output.
    """
    roles = parse_roles_with_bullets(resume_text)
    if not roles:
        return ""

    out = [
        "══════════════════════════════════════════════════════════════",
        "WORK EXPERIENCE SKELETON — COMPLETE EVERY ◆SLOT",
        "══════════════════════════════════════════════════════════════",
        "Use this skeleton for the Work Experience section of your output.",
        "Replace each ◆SLOT-N with exactly one rewritten bullet.",
        "Rules:",
        "  • One ◆SLOT-N → one output bullet.  Never merge two slots.",
        "  • Never delete a ◆SLOT line.",
        "  • You MAY add extra bullets after the last ◆SLOT for each role.",
        "  • Rewrite every slot in stronger, sector-targeted language.",
        "",
    ]

    for heading, bullets in roles:
        out.append(f"**{heading}**")
        for i, b in enumerate(bullets, 1):
            out.append(f"◆SLOT-{i}: [rewrite of → \"{b}\"]")
        out.append("[add extra bullets here if the role/JD warrants it]")
        out.append("")

    out.append("══════════════════════════════════════════════════════════════")
    return "\n".join(out)


def count_output_bullets(text):
    """
    Count bullets (lines starting with – or - or • or *) per role heading
    in a markdown-formatted resume output.  Returns {heading: count}.
    """
    bullet_re = re.compile(r'^\s*[-–•*]\s')
    heading_re = re.compile(r'^\s*\*\*(.+?)\*\*')
    lines = text.splitlines()
    counts = {}
    current = None
    for line in lines:
        h = heading_re.match(line)
        if h:
            current = h.group(1).strip()
            if current not in counts:
                counts[current] = 0
        elif bullet_re.match(line) and current:
            counts[current] += 1
    return counts


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
    skeleton = build_we_skeleton(base_resume)
    roles_parsed = parse_roles_with_bullets(base_resume)
    # Build required minimums dict keyed by role heading for self-correction
    minimums = {heading: len(bullets) for heading, bullets in roles_parsed}

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

{skeleton}

Please rewrite my resume following all the rules in your instructions. Tailor it for the {role_label} role at a {sector_label} company.

IMPORTANT: For the Work Experience section, use the skeleton above as your template. Fill in every ◆SLOT-N with a rewritten bullet. Do not skip, merge, or delete any ◆SLOT line."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable is not set."}, 500

    def generate():
        client = OpenAI(api_key=api_key)
        try:
            # ── Pass 1: generate full resume (non-streaming so we can inspect) ──
            resp1 = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=8000,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=False,
            )
            draft = resp1.choices[0].message.content

            # ── Pass 2: self-correction if any role is under the minimum ─────
            output_counts = count_output_bullets(draft)
            shortfalls = []
            for heading, minimum in minimums.items():
                # find the best-matching heading key in the output
                matched_count = 0
                for out_heading, cnt in output_counts.items():
                    # fuzzy match: check if key words from the parsed heading
                    # appear in the output heading
                    key_words = [w for w in heading.split() if len(w) > 3
                                 and w.lower() not in ('london', 'india', 'hyderabad')]
                    if any(kw.lower() in out_heading.lower() for kw in key_words):
                        matched_count = cnt
                        break
                if matched_count < minimum:
                    shortfalls.append(
                        f"- {heading}: has {matched_count} bullets, needs {minimum}"
                    )

            final = draft
            if shortfalls:
                fix_prompt = (
                    "The resume draft below has fewer bullets than required in some roles.\n"
                    "Expand ONLY the under-populated roles listed here (do not change anything else):\n\n"
                    + "\n".join(shortfalls)
                    + "\n\nFor each role listed, add bullets drawn from the candidate's real "
                    "experience until it meets or exceeds the required minimum. "
                    "Return the complete corrected resume.\n\n"
                    "DRAFT:\n" + draft
                )
                resp2 = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=8000,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": fix_prompt},
                    ],
                    stream=False,
                )
                final = resp2.choices[0].message.content

            # ── Stream the final output to the client ────────────────────────
            # Emit in chunks so the frontend streaming logic still works
            chunk_size = 120
            for i in range(0, len(final), chunk_size):
                yield f"data: {json.dumps({'text': final[i:i+chunk_size]})}\n\n"

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
