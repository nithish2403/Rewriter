import os
import json
import io
import pdfplumber
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

5. NO BUZZWORD STACKING: Every skill or tool must appear in the context of actual work. The following phrases are BANNED anywhere in the resume — do not use them under any circumstances:
   • "proven track record" • "extensive experience" • "adept at" • "passionate about"
   • "excellent communicator" • "team player" • "results-driven" • "detail-oriented"
   • "dynamic" • "innovative" • "leverage" • "synergy" • "best-in-class"
   • "strong background in" • "deep expertise in" (in the summary) • "seasoned"
   If you find yourself writing any of these, stop and replace with a concrete evidence statement.

6. BULLET DISCIPLINE: Strong active verbs. Outcome-focused. Past tense for previous roles, present tense for current role. Bullets should be 1–2 lines — but "1–2 lines" means CONTENT-RICH lines, not stripped-down summaries. For senior technical roles a 30–50 word bullet with specific tools, methods, and outcomes is correct. Do NOT condense a rich original bullet into a vague one-liner. Preserve all named technologies, tools, metrics, and technical specifics from the original. Depth and specificity make bullets credible; vagueness makes them weak.

8. VERB VARIETY — STRICT NO REPEATS: Every action verb across the entire resume must be UNIQUE — no verb may appear more than ONCE as the opening word of any bullet. Before writing each bullet, check every verb already used. If the verb you are about to use has already appeared, pick a different one.
   Commonly overused verbs to actively avoid repeating: automated, designed, developed, implemented, managed, created, built, delivered, established, led, drove, improved, optimised, streamlined, maintained, deployed.
   Verb bank — pick from this list for variety:
   — Building/creating: architected, engineered, constructed, provisioned, authored, shipped, launched, forged, assembled, spun up, rolled out
   — Improving: accelerated, elevated, hardened, tightened, reduced, cut, trimmed, boosted, reinforced, consolidated, refined, overhauled, rationalised
   — Owning/running: operated, owned, stewarded, administered, oversaw, governed, directed, ran, coordinated
   — Analysing/solving: diagnosed, resolved, investigated, profiled, audited, surfaced, instrumented, modelled, uncovered, quantified
   — Enabling/supporting: enabled, unblocked, standardised, codified, onboarded, championed, embedded, facilitated
   — Designing/planning: scoped, specified, structured, mapped, orchestrated, planned, framed, blueprinted
   — Integrating/connecting: wired, integrated, bridged, unified, connected, federated
   After drafting all bullets, do a final scan: list every opening verb. If any appears twice, revise before submitting.

9. IMPACT BULLET — ONE PER COMPANY: The first bullet under every company must follow this exact narrative structure — all within a single cohesive 2–3 sentence bullet (not a list):
   STRUCTURE: [Problem context: what was broken, missing, or at risk] → [Why it mattered: business cost, customer pain, operational risk, or compliance exposure] → [What you owned and how you solved it: your specific role, tools, and approach] → [Quantified outcome: metric, benchmark, or before/after comparison] → [Downstream impact: what changed in the team, system, or business as a result]
   EXAMPLE FORMAT: "Infrastructure lacked repeatable account provisioning across 7 AWS accounts, creating inconsistent security postures and slowing onboarding; owned end-to-end design and delivery of a CloudFormation-based multi-account baseline that reduced provisioning time from hours to 6 minutes, standardised logging and IAM controls across all accounts, and eliminated a recurring class of security audit findings."
   This bullet must be the richest, most narrative entry for that company. It sets context for everything that follows. Label it in the skeleton as ◆IMPACT.

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
• Only use figures explicitly present in the original resume. Do not invent numbers.
• Remove or soften anything that sounds inflated or difficult to defend in an interview.

• METRIC TYPE DIVERSITY — CRITICAL: Do NOT rely on percentages alone. Across the full resume, use a deliberate mix of all of these metric types:
  — Percentage-based: "reduced errors by 15%", "improved throughput by 40%"
  — Absolute financial: "saved £30k/year", "eliminated £10k/month in avoidable spend"
  — Time-based: "cut provisioning from hours to 6 minutes", "reduced mean-time-to-detect by 35%", "delivered in 3 weeks"
  — Scale/count-based: "across 7 AWS accounts", "supporting 250+ users", "1M+ assets processed", "15+ engineers"
  — Ranking/positioning: "first team to achieve X", "reduced to zero incidents in Q1", "consistently met 99.9% SLA"
  — Scope statements (where no metric exists): "owned end-to-end", "sole engineer responsible for", "across 3 environments"
  No more than 40% of metric-bearing bullets should use percentages. If you have written 4 bullets with % metrics, the next metric must be a different type.

• WORD FREQUENCY — avoid repeating high-frequency generic words across bullets:
  — Instead of "AWS" every time: use "the platform", "cloud environments", "the multi-account estate", "our AWS estate"
  — Instead of "infrastructure" every time: use "platform", "environment", "estate", "systems", "stack"
  — Instead of "reduce/reduced" every time: use "cut", "trimmed", "brought down", "halved", "shrunk", "lowered"
  — Instead of "improve/improved" every time: use "accelerated", "elevated", "tightened", "boosted", "sharpened"
  No single common word (aws, infrastructure, reduce, improve, implement, automate) should appear more than 3 times across the full resume text.

• ABBREVIATION HYGIENE: Never write AWS service names as bare isolated abbreviations in a sentence. Write them in context:
  — WRONG: "Used S3, EC2, IAM to manage..."
  — RIGHT: "Managed access controls via IAM Identity Center, compute via EC2, and object storage in Amazon S3..."
  Spell out what a service does on first mention if the audience may not know it. Never leave single-letter words (S, a, E) floating in a bullet from formatting artefacts.

• INDUSTRY TERMINOLOGY: Each bullet should use at least one piece of role-specific technical vocabulary drawn from the job description or sector. Generic verbs with no technical grounding are weak. E.g. for a DevOps/SRE role: reference SLOs, toil reduction, runbook automation, blast radius, change failure rate, MTTR, canary deployments, GitOps, shift-left testing, observability signals, pager rota — where the candidate's experience genuinely supports it.

═══════════════════════════════════════════════════════════
THEMES TO EMPHASISE (where supported by real experience)
═══════════════════════════════════════════════════════════

Core: AWS infrastructure, multi-account environments, Infrastructure as Code, CI/CD, Linux, Python automation, IAM / access control / security, observability and monitoring, incident response, reliability, resilience, cost optimisation, standardisation, repeatable platform delivery.

═══════════════════════════════════════════════════════════
SECTOR-SPECIFIC EMPHASIS
═══════════════════════════════════════════════════════════

The target sector mandate is provided in the user message. Apply it precisely to every bullet.
Do not blend in themes from other sectors — focus exclusively on the sector stated in the mandate.

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
        out.append(
            "◆IMPACT: [Write the IMPACT BULLET here — see Rule 9. "
            "2–3 sentences covering: problem context → why it mattered → "
            "your ownership & approach → quantified outcome → downstream impact. "
            f"Draw from these originals for raw material: {'; '.join(bullets[:3])}]"
        )
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

# ── Sector-specific emphasis injected dynamically into the user message ───────
# Only the relevant sector's block is sent — model doesn't have to choose.

SECTOR_EMPHASIS = {
    "investment_banking": """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET SECTOR MANDATE — INVESTMENT BANKING / HEDGE FUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This resume must read like it belongs at Goldman Sachs, JP Morgan, Citadel, or a top-tier quant fund.
Every bullet must lead with or prominently feature at least one of these themes:
  → Security posture and access governance (IAM, PAM, least-privilege, audit trail)
  → Change management and controlled deployments (CAB, four-eyes approval, production freeze)
  → Operational risk reduction (blast radius containment, rollback procedures, incident playbooks)
  → Regulatory and audit readiness (SOX alignment, immutable logs, traceability)
  → Disaster recovery and resilience (RTO/RPO, DR runbook, BCP testing, multi-region failover)
  → Production reliability (MTTR, change failure rate, SLA adherence, zero-downtime deployments)

Vocabulary to weave in naturally (only where the experience genuinely supports it):
  change advisory board (CAB), privileged access management (PAM), immutable audit trail,
  four-eyes approval, production freeze, DR runbook, RTO/RPO, change failure rate,
  mean-time-to-restore (MTTR), regulatory change control, access recertification,
  operational risk framework, segregation of duties.

Tone: senior, measured, risk-aware. Every engineering decision had a governance reason.
""",
    "fintech": """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET SECTOR MANDATE — FINTECH STARTUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This resume must read like it belongs at Monzo, Wise, Revolut, Stripe, GoCardless, or a Series B/C fintech.
Every bullet must lead with or prominently feature at least one of these themes:
  → Delivery velocity and CI/CD pipeline maturity (trunk-based dev, feature flags, progressive delivery)
  → Developer experience and platform self-service (golden path, internal developer platform)
  → Scalability and cost efficiency (FinOps, horizontal scaling, cloud-native architecture)
  → Observability and reliability (SLO/error budget, on-call, incident retrospective, MTTR)
  → Rapid iteration and pragmatic engineering (shipped fast, unblocked teams, reduced toil)

Vocabulary to weave in naturally (only where the experience genuinely supports it):
  trunk-based development, feature flags, progressive delivery, platform engineering,
  golden path, SLO/error budget, on-call rota, incident retrospective, developer productivity,
  cloud-native, FinOps, self-service, toil reduction, observability signals.

Tone: fast-moving, pragmatic, engineer-led. You shipped things that mattered and kept them running.
""",
    "big_tech": """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET SECTOR MANDATE — BIG TECH (FAANG-TIER)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This resume must read like it belongs at Google, AWS, Meta, Apple, or Microsoft.
Every bullet must lead with or prominently feature at least one of these themes:
  → Systems thinking and scale (millions of requests, petabytes of data, global distribution)
  → Observability depth (distributed tracing, SLI/SLO/error budget, anomaly detection)
  → Automation rigour and IaC maturity (no manual toil, everything codified, self-healing systems)
  → Incident response and reliability (blameless post-mortem, blast radius, on-call escalation)
  → Distributed systems design (dependency graph, capacity planning, fault tolerance)

Vocabulary to weave in naturally (only where the experience genuinely supports it):
  SLO/SLI/error budget, toil reduction, blameless post-mortem, canary deployment,
  blast radius, runbook automation, distributed tracing, capacity planning,
  oncall escalation, dependency graph, correctness at scale.

Tone: rigorous, systems-level, principled. Every architectural decision had a reason at scale.
""",
    "enterprise": """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET SECTOR MANDATE — TRADITIONAL ENTERPRISE TECH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This resume must read like it belongs at a large insurance firm, enterprise software vendor, or FTSE 100 IT function.
Every bullet must lead with or prominently feature at least one of these themes:
  → Stability and standardisation (repeatable processes, configuration baselines, platform consistency)
  → Multi-team coordination and stakeholder management (programme delivery, cross-functional alignment)
  → Governance and documentation (service catalogue, CMDB, runbooks, knowledge base)
  → Change control and compliance (ITIL-aligned processes, CAB, audit readiness)
  → Business continuity and capacity management (BCP, SLA/OLA, service management)

Vocabulary to weave in naturally (only where the experience genuinely supports it):
  ITIL, CMDB, change advisory board, service catalogue, SLA/OLA,
  capacity management, business continuity, programme delivery,
  centre of excellence (CoE), service management, configuration baseline.

Tone: structured, collaborative, governance-aware. You brought order, repeatability, and cross-team alignment.
""",
    "banking": """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET SECTOR MANDATE — BANKING (RETAIL / COMMERCIAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This resume must read like it belongs at Barclays, HSBC, Lloyds, NatWest, Santander, or similar.
Every bullet must lead with or prominently feature at least one of these themes:
  → Regulatory awareness and compliance (PRA/FCA alignment, DORA metrics, operational resilience)
  → Secure platform delivery (access governance, privileged access, audit readiness)
  → Production stability (zero-downtime deployments, production freeze adherence, change control)
  → Operational resilience (BCP/DR testing, RTO/RPO, incident management, on-call)
  → Access governance (access recertification, least-privilege, PAM, identity lifecycle)

Vocabulary to weave in naturally (only where the experience genuinely supports it):
  PRA/FCA alignment, operational resilience framework, DORA metrics,
  access recertification, privileged access management, production freeze,
  audit readiness, BCP/DR testing, change advisory board, segregation of duties.

Tone: risk-aware, compliance-conscious, operationally rigorous. Every deployment was controlled and accountable.
""",
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
    sector_emphasis = SECTOR_EMPHASIS.get(sector_key, "")
    skeleton = build_we_skeleton(base_resume)
    roles_parsed = parse_roles_with_bullets(base_resume)
    # Build required minimums dict keyed by role heading for self-correction
    minimums = {heading: len(bullets) for heading, bullets in roles_parsed}

    user_message = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — ANALYSE THE JOB DESCRIPTION BEFORE WRITING ANYTHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the job description below carefully and extract:
  A. The top 6–8 TECHNICAL requirements / themes this role demands day-to-day
  B. The SENIORITY SIGNALS — what does "mid-senior {role_label}" look like in this JD?
     What ownership level, scope of impact, and technical depth does it expect?
  C. The 10–15 most important KEYWORDS for ATS (tools, frameworks, methodologies named in the JD)
  D. The TERMINOLOGY AND VOCABULARY this specific role and sector uses
     (the exact words a hiring manager at this company would use)

Use A–D as the PRIMARY LENS for every bullet you rewrite in Step 2.
Every rewritten bullet must speak directly to what this JD is asking for.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — REWRITE THE RESUME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGET ROLE: {role_label}
TARGET SECTOR: {sector_label}
{sector_emphasis}
══════════════════════════════
BASE RESUME
══════════════════════════════
{base_resume}

══════════════════════════════
JOB DESCRIPTION
══════════════════════════════
{job_description}

{skeleton}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BULLET REWRITE MANDATE — READ THIS BEFORE FILLING ANY ◆SLOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each ◆SLOT contains an ORIGINAL bullet as raw material — not a template to lightly rephrase.

For EVERY slot, follow this process:
  1. Identify which JD requirement (from your Step 1 analysis) this experience speaks to
  2. Reframe the bullet so that connection is explicit — use the JD's own vocabulary
  3. Write it at the voice and depth of a mid-senior {role_label} — show ownership, scope, technical judgement
  4. Use the sector vocabulary from the SECTOR MANDATE above where the experience genuinely supports it

A recruiter reading each bullet must immediately think: "this person has done exactly what we need."

Do NOT lightly rephrase the original. The substance stays grounded in the candidate's real experience,
but the framing, vocabulary, and emphasis must shift to match this specific role and sector.

Please rewrite the resume following all rules in your instructions.

IMPORTANT: For the Work Experience section, use the skeleton above as your template.
Fill in every ◆SLOT-N with a rewritten bullet. Do not skip, merge, or delete any ◆SLOT line."""

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
                temperature=0.4,
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

            # ── Detect repeated verbs in draft ───────────────────────────────
            watch_verbs = [
                "automated","designed","developed","implemented","managed",
                "created","built","delivered","established","led","drove",
                "improved","optimised","streamlined","maintained","deployed",
            ]
            bullet_lines = [l.strip() for l in draft.splitlines()
                            if re.match(r'^\s*[◆•\-–]\s', l)]
            verb_counts = {}
            for bl in bullet_lines:
                first_word = bl.lstrip("◆•-– ").split()[0].lower().rstrip(",") if bl.lstrip("◆•-– ").split() else ""
                if first_word in watch_verbs:
                    verb_counts[first_word] = verb_counts.get(first_word, 0) + 1
            repeated_verbs = {v: c for v, c in verb_counts.items() if c > 1}

            # ── Detect high-frequency repeated words anywhere in bullets ─────
            # Rule: no common word should appear more than 3 times across
            # the full resume (matches the WORD FREQUENCY rule in SYSTEM_PROMPT).
            HIGH_FREQ_WATCH = {
                # reduce family
                "reducing", "reduced", "reduce",
                # improve family
                "improving", "improved", "improve",
                # implement family
                "implementing", "implemented", "implement",
                # increase family
                "increasing", "increased", "increase",
                # manage family
                "managing", "managed", "manage",
                # ensure family
                "ensuring", "ensured", "ensure",
                # utilise family
                "utilising", "utilised", "utilise",
                "utilizing", "utilized", "utilize",
                # support family
                "supporting", "supported", "support",
                # enable family
                "enabling", "enabled", "enable",
                # create family
                "creating", "created", "create",
                # build family
                "building", "built", "build",
                # deliver family
                "delivering", "delivered", "deliver",
                # maintain family
                "maintaining", "maintained", "maintain",
            }
            all_bullet_text = " ".join(bullet_lines).lower()
            repeated_words = {}
            for word in HIGH_FREQ_WATCH:
                count = len(re.findall(r'\b' + re.escape(word) + r'\b', all_bullet_text))
                if count > 3:
                    repeated_words[word] = count

            fix_parts = []
            if shortfalls:
                fix_parts.append(
                    "ISSUE 1 — BULLET COUNT: The following roles need more bullets:\n"
                    + "\n".join(shortfalls)
                    + "\nAdd bullets from the candidate's real experience until each meets its minimum."
                )
            if repeated_verbs:
                verb_list = ", ".join(f'"{v}" ({c}x)' for v, c in repeated_verbs.items())
                fix_parts.append(
                    f"ISSUE 2 — REPEATED OPENING VERBS: These action verbs open more than one bullet: {verb_list}. "
                    "Every opening verb across ALL bullets must be unique — no verb may appear more than once. "
                    "Replace duplicate uses with varied alternatives from the verb bank in your instructions."
                )
            if repeated_words:
                word_list = ", ".join(
                    f'"{w}" ({c}x)'
                    for w, c in sorted(repeated_words.items(), key=lambda x: -x[1])
                )
                fix_parts.append(
                    f"ISSUE 3 — REPEATED WORDS (mid-sentence): These common words appear more than 3 times "
                    f"across all bullet text: {word_list}. "
                    "The rule states no single common word should appear more than 3 times across the full resume. "
                    "Replace excess uses with varied synonyms. Synonym guide:\n"
                    "  • reduce/reducing/reduced → cut, trimmed, brought down, halved, shrunk, lowered, decreased, curtailed\n"
                    "  • improve/improving/improved → accelerated, elevated, tightened, boosted, sharpened, enhanced, strengthened\n"
                    "  • implement/implementing/implemented → deployed, rolled out, shipped, launched, engineered, introduced, constructed\n"
                    "  • manage/managing/managed → operated, oversaw, stewarded, owned, governed, directed, coordinated\n"
                    "  • ensure/ensuring/ensured → enforced, guaranteed, validated, hardened, cemented\n"
                    "  • utilise/utilising/utilised → applied, adopted, harnessed, via [as preposition]\n"
                    "  • support/supporting/supported → underpinned, sustained, reinforced, backed\n"
                    "  • enable/enabling/enabled → unblocked, empowered, facilitated, unlocked\n"
                    "  • maintain/maintaining/maintained → sustained, operated, kept, governed, stewarded\n"
                    "Rewrite the affected bullets so each word's total count across the resume is 3 or below. "
                    "Return the full corrected resume."
                )

            final = draft
            if fix_parts:
                fix_prompt = (
                    "Fix the following issues in the resume draft below. "
                    "Return the complete corrected resume.\n\n"
                    + "\n\n".join(fix_parts)
                    + "\n\nDRAFT:\n" + draft
                )
                resp2 = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=8000,
                    temperature=0.4,
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


VALIDATE_SYSTEM_PROMPT = """You are a senior resume quality auditor and ATS (Applicant Tracking System) specialist. You have deep knowledge of how modern ATS platforms (Workday, Greenhouse, Lever, iCIMS, Taleo, SmartRecruiters) parse and rank resumes, and what hiring managers at technical companies look for.

Analyse the resume text provided and return a comprehensive validation report as valid JSON.

Check ALL of the following categories thoroughly:

1. GRAMMAR & LANGUAGE
   - Grammatical errors (subject-verb agreement, article usage, prepositions)
   - Spelling mistakes and typos
   - Punctuation errors (missing full stops, inconsistent comma usage)
   - Tense inconsistency (current role = present tense, past roles = past tense)
   - Sentence fragments or run-on sentences
   - British vs American spelling inconsistency

2. FORMATTING & WHITESPACE
   - Double or extra spaces between words
   - Inconsistent bullet point styles (mixing -, •, *, –, ◆ or any combination of these characters)
     IMPORTANT: When flagging this issue, you MUST quote the exact bullet text where the non-standard
     character appears (e.g. '◆ Architected a multi-account...') and name the company/role it is under.
     The location field must be specific, e.g. "Konsistent Consulting, bullet 1: '◆ Architected...'"
     NOT just "Work Experience section". If multiple bullets use the non-standard character, quote all of them.
   - Inconsistent date formats (mixing "Jan 2024" with "01/2024" etc.)
   - Inconsistent capitalisation in headings or job titles
   - Trailing spaces or blank lines within sections
   - Missing space after bullet character

3. ATS COMPATIBILITY
   - Contact info completeness: name, email, phone, LinkedIn/GitHub
   - Personal discriminatory info present (photo, DOB, nationality, marital status, religion) — flag these as errors
   - Standard section headings used (Experience/Work Experience, Education, Skills, Certifications) — non-standard headings fail ATS
   - Skills section present and parseable
   - All job entries have: company name, job title, dates, location
   - Date format is ATS-parseable (Month Year or MM/YYYY — not just years alone)
   - No tables, text boxes, or columns inferred from formatting
   - File is likely single-column layout (infer from text order)
   - Keywords: if a job description was provided, flag important JD keywords missing from the resume

4. CONTENT QUALITY
   - Buzzwords and clichés present (proven track record, extensive experience, adept at, results-driven, passionate, synergy, leverage, innovative, dynamic, team player, detail-oriented)
   - Passive voice overuse ("was responsible for" instead of active verbs)
   - Vague statements with no evidence ("worked on various projects", "helped with...")
   - Quantified achievements: flag any role that has NO metrics at all
   - Action verb diversity: flag if same opening verb used 2+ times across bullets
   - Word repetition (mid-sentence): scan ALL bullet text for any single word (e.g. "reducing", "improving",
     "managing", "ensuring", "utilising", "supporting") that appears more than 3 times across the full resume.
     Flag each repeated word and quote 2–3 examples of the bullets where it appears so the candidate can see
     the pattern clearly. Suggest specific synonyms (e.g. "reducing" → cut, trimmed, lowered, curtailed).
   - Impact statements: are outcomes clear, or just task descriptions?

5. STRUCTURAL COMPLETENESS
   - Professional summary or headline present
   - All roles have start AND end dates (or "Present" for current)
   - Employment gaps > 6 months (flag for candidate awareness — not necessarily an error)
   - Education section present
   - Certifications listed if relevant to technical role
   - Consistent company → job title → date → location order across all roles

6. LENGTH & DENSITY
   - Resume length: warn if likely > 2 pages for < 5 years experience, or < 1 page for > 5 years
   - Bullet count per role:
     → < 3 bullets = too thin (flag as warning)
     → 10 bullets = borderline dense (flag as info only)
     → > 10 bullets = too dense (flag as warning)
     → 4–9 bullets per role is the healthy range — DO NOT flag this as dense, even for a current role.
     → A current senior/mid-senior role with 7–9 bullets is CORRECT and should NOT be flagged.
     → Only flag density if bullet count is genuinely over 10.
   - Summary length: warn if > 6 lines (too long for ATS snippet)

Return ONLY valid JSON — no markdown, no code fences, no commentary outside the JSON. Use this exact schema:

{
  "overall_score": <integer 0-100>,
  "overall_grade": <"A" | "B" | "C" | "D" | "F">,
  "summary": "<2-3 sentence overall assessment>",
  "ats_compatibility_score": <integer 0-100>,
  "ready_to_submit": <true | false>,
  "categories": [
    {
      "id": "<snake_case_id>",
      "name": "<Display Name>",
      "score": <integer 0-100>,
      "status": <"pass" | "warn" | "fail">,
      "issues": [
        {
          "severity": <"error" | "warning" | "info">,
          "text": "<clear description of the issue>",
          "location": "<where in the resume, e.g. 'Konsistent Consulting, bullet 3' or 'Summary section'>",
          "suggestion": "<specific actionable fix>"
        }
      ]
    }
  ]
}

Severity guide:
- "error": must fix before submitting (grammar error, missing contact info, discriminatory info, ATS-breaking format)
- "warning": should fix, will hurt score (vague bullet, missing metric, repeated verb)
- "info": nice to improve, minor issue (style suggestion, optional enhancement)

Be specific and actionable. Quote the exact text that has an issue where possible."""


@app.route("/validate", methods=["POST"])
def validate():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable is not set."}, 500

    pdf_file = request.files.get("pdf")
    job_desc = (request.form.get("job_description") or "").strip()

    if not pdf_file:
        return {"error": "No PDF file uploaded."}, 400

    # ── Extract text from PDF ─────────────────────────────────────────────────
    try:
        pdf_bytes = pdf_file.read()
        resume_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=3)
                if page_text:
                    resume_text += page_text + "\n\n"
        resume_text = resume_text.strip()
    except Exception as e:
        return {"error": f"Could not read PDF: {str(e)}"}, 400

    if not resume_text:
        return {"error": "Could not extract text from PDF. Make sure it is a text-based PDF, not a scanned image."}, 400

    # ── Build user message ────────────────────────────────────────────────────
    user_msg = f"RESUME TO VALIDATE:\n\n{resume_text}"
    if job_desc:
        user_msg += f"\n\n---\nJOB DESCRIPTION (for keyword gap analysis):\n\n{job_desc}"

    # ── Call GPT-4o ───────────────────────────────────────────────────────────
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4000,
            temperature=0,
            messages=[
                {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        report = json.loads(raw)
        return report, 200

    except AuthenticationError:
        return {"error": "Invalid API key."}, 401
    except RateLimitError:
        return {"error": "Rate limit reached. Please wait and try again."}, 429
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}, 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
