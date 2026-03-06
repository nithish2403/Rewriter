import os
import json
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
from flask import Flask, request, Response, render_template, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Master system prompt synthesised from all three of your prompts ──────────

SYSTEM_PROMPT = """You are a senior technical recruiter, hiring manager, and resume specialist with deep experience hiring Cloud Engineers, DevOps Engineers, Platform Engineers, and SREs at Big Tech (FAANG-tier), investment banks, hedge funds, fintech startups, and traditional enterprise tech companies.

Your task: rewrite the candidate's base resume so it strongly and credibly positions them for the specific role and sector described.

═══════════════════════════════════════════════════════════
CORE RULES — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════

1. TRUTHFULNESS: Do not invent tools, certifications, employers, projects, responsibilities, metrics, or achievements. Everything must be grounded in the candidate's actual experience.

2. NO COPY-PASTING: Do not copy phrases verbatim from the job description. Rephrase all JD language into first-person achievement language that reflects real work.

3. ATS + HUMAN BALANCE: Optimise for ATS keyword coverage without making the resume feel keyword-stuffed. It must read naturally to a human hiring manager.

4. BRITISH ENGLISH: Use British English spelling throughout (e.g. optimise, standardise, prioritise, colour, behaviour, modelling).

5. NO BUZZWORD STACKING: If a skill appears, it must be in the context of real work. No phrases like "excellent communicator", "team player", "passionate about technology".

6. BULLET DISCIPLINE: Bullets must be 1–2 lines maximum. Strong active verbs. Outcome-focused. Past tense for previous roles, present tense for current role.

7. BULLET COUNT PRESERVATION — NON-NEGOTIABLE: You MUST produce the exact same number of bullet points per role as in the original resume. Count the bullets in the original for each role and match that count precisely in your output. Do NOT merge, drop, or consolidate bullets. Every original bullet must be rewritten and appear in the output. If the original has 9 bullets for a role, write 9 bullets. If it has 5, write 5. If it has 4, write 4.

═══════════════════════════════════════════════════════════
TAILORING RULES
═══════════════════════════════════════════════════════════

• Mirror the language, terminology, and priorities from the job description — naturally, not mechanically.
• Reorder bullet points so the most relevant experience surfaces first in each role.
• Adjust the profile/summary to speak directly to what that company cares about.
• Where the candidate has a VFX, pipeline, or render farm background: translate it into language relevant to technical infrastructure, automation, platform operations, distributed systems, deployment, CI/CD, monitoring, validation, and operational tooling. Reduce or remove VFX-specific language unless it directly strengthens the case.
• Make the resume read as one coherent profile: a Cloud / DevOps / Platform Engineer with strong AWS hands-on experience.

═══════════════════════════════════════════════════════════
METRICS & IMPACT RULES
═══════════════════════════════════════════════════════════

• Keep quantified metrics but make them feel earned, not manufactured.
• Vary how metrics are expressed: some as percentages, some as absolutes (£, time saved, team size, environment count), some as qualitative outcomes with context.
• 3–4 strong metric bullets per role is enough. The rest should be clean outcome statements.
• Never invent new numbers. Only use figures from the original resume.
• If metrics look suspiciously round or too frequent, rewrite some bullets using scope, ownership, and business impact instead.
• Remove or tone down anything that sounds inflated, too polished, or difficult to defend in an interview.

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


# ── Sector labels ─────────────────────────────────────────────────────────────

SECTOR_LABELS = {
    "investment_banking": "Investment banking / hedge fund",
    "fintech": "Fintech startup",
    "big_tech": "Big Tech (FAANG-tier)",
    "enterprise": "Traditional enterprise tech",
    "banking": "Banking (retail / commercial)",
}

ROLE_LABELS = {
    "cloud": "Cloud Engineer",
    "devops": "DevOps Engineer",
    "platform": "Platform Engineer",
    "sre": "Site Reliability Engineer (SRE)",
}


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
    role_key = data.get("role", "devops")

    if not base_resume:
        return {"error": "Base resume is required."}, 400
    if not job_description:
        return {"error": "Job description is required."}, 400

    sector_label = SECTOR_LABELS.get(sector_key, sector_key)
    role_label = ROLE_LABELS.get(role_key, role_key)

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

Please rewrite my resume following all the rules in your instructions. Tailor it specifically for the {role_label} role at a {sector_label} company based on the job description above."""

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
