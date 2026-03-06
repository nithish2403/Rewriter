#!/usr/bin/env python3
"""
Diagnose bullet counting and model output for the actual base resume.
Run: OPENAI_API_KEY=sk-... python3 test_bullets.py
"""
import os, re, json
from openai import OpenAI

# ── paste resume ─────────────────────────────────────────────────────────────

BASE_RESUME = """Cloud Engineer (AWS) specialising in multi-account operations, infrastructure automation, and secure platform delivery. Experienced operating AWS environments supporting 250+ users across 7 AWS accounts, delivering repeatable baselines, data transfer/archival pipelines, and cost-optimised compute strategies. Strong hands-on skills in IaC (CloudFormation/Terraform), Linux, Python automation, networking fundamentals (VPC), and observability to improve reliability and operational response.
Key Skills
AWS: EC2, IAM, S3, VPC, CloudWatch, CloudTrail, Secrets Manager, IAM Identity Center, Route 53, KMS.
IaC: CloudFormation, Terraform, YAML, JSON.
Container: Docker, Kubernetes, ECS, EKS.
Automation: Python, Bash, PowerShell, REST APIs, Ansible.
CI/CD & Version Control: GitHub Actions, GitLab CI, Git, Jira, Confluence.
WORK EXPERIENCE
Konsistent Consulting (AWS Partner)								             London, UK
Cloud Engineer                                         			                                                                           January 2025 - Present
* Operated 7 AWS accounts and customer facing services for 250+ users, owned access controls (IAM/Identity Center), compute (EC2), storage (S3) and networking (VPC) with CloudWatch telemetry and CloudTrail audit trails.
* Implemented an internal AI chat assistant for AWS Deadline Cloud by implementing an MCP server from scratch, enabling fast self-serve visibility into render farms and speeding up incident triage and capacity decisions during peak demand.
* Delivered a versioned multi-account baseline in 6 minutes using CloudFormation (with targeted Terraform guardrails), standardising logging, security posture, and account setup for repeatable delivery.
* Developed a real-time capacity and allocation visibility tool across regions/AZs to reduce 'insufficient capacity' blockers by 40% and improve provisioning efficiency during peak demand.
* Eliminated £10k/month avoidable spend via Spot-first scaling, rightsizing, lifecycle policies, and resource governance, drove structured cost reviews and remediation using utilisation signals.
* Designed large-scale data transfer and archival workflows to S3 from on-prem storage with Python validation and deduplication, increased backup efficiency 70% and saved £30k/year through lifecycle tiering.
* Established CI-driven build and release pipelines (GitHub Actions) to build/test/publish versioned runtime packages; lowered environment failures 80% through dependency pinning and smoke test coverage.
* Standardised isolated execution for packages using Docker and Kubernetes (EKS) to run repeatable validation workloads and minimise drift across environments.
Milk VFX   									                                            London, UK
Render Wrangler / Developer                                         			                                                   March 2024 - January 2025
* Streamlined deployment and configuration tasks for distributed compute workloads using Python/Bash, reducing manual operational overhead and improving repeatability.
* Hardened job submission and validation tooling with guardrails and configuration checks, increasing first-pass success rates and reducing support tickets (25% improvement in error resolution).
* Implemented operational dashboards and alerting using CloudWatch/Grafana and custom scripts, accelerating incident detection and reducing time-to-detect by 35% for common failure modes.
* Diagnosed and resolved infrastructure and application issues in production, improving utilisation (20%) and reducing downtime during delivery windows.
* Partnered with data analysts to extract workload level stats from platform reports and logs to guide optimisation and capacity planning.
Ghost VFX   									                                            London, UK
Pipeline Developer	                                        			                                             February 2023 - February 2024
* Created automation tools that processed and validated 1M+ assets/work items with integrity checks, lowering errors by 15% through validation and guardrails.
* Released internal tools with structured versioning, documentation, testing, and regression checks, cutting deployment errors by 25%.
* Maintained GitLab CI pipelines with automated testing, packaging, and controlled deployments, increasing release reliability by 40%.
* Designed user-friendly tooling interfaces, improving workflow navigation for 15+ users and reducing task completion times by 40%.
Pixelloid Studios                                                                                                                                                                  Hyderabad, India
VFX Developer		                                         			                                                               July 2019 - July 2021
* Developed procedural tooling to generate 25+ variants of assets/workflows, reducing creation time by 30% through automation and reusable templates.
CERTIFICATIONS
* AWS Certified Cloud Practitioner
* AWS Solutions Architect Associate"""

JOB_DESCRIPTION = """QA Automation Engineer | Expertise in Automation Frameworks and Test Strategy
Proven QA Automation Engineer with a robust background in building and maintaining scalable test frameworks and CI/CD pipelines, committed to delivering high-quality, reliable software solutions. Deep expertise in Java, Selenium WebDriver, and API testing complemented by strong skills in cloud environments, including AWS. Adept at collaborating within Agile teams to enhance QA processes and drive digital transformation in traditional enterprise settings."""


# ── Bullet counter (same logic as app.py) ────────────────────────────────────

def count_bullets_per_role(resume_text):
    bullet_pattern = re.compile(r'^\s*[-•·*]')
    lines = resume_text.splitlines()
    counts = {}
    current_heading = None
    current_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if bullet_pattern.match(line):
            current_count += 1
        else:
            if current_heading and current_count > 0:
                counts[current_heading] = current_count
            current_heading = stripped[:80]
            current_count = 0
    if current_heading and current_count > 0:
        counts[current_heading] = current_count
    return counts


def build_bullet_constraints(resume_text):
    counts = count_bullets_per_role(resume_text)
    if not counts:
        return "(none detected)"
    lines = ["MANDATORY BULLET COUNTS — you MUST produce AT LEAST this many bullets for each role:"]
    for heading, count in counts.items():
        lines.append(f"  • {heading}: minimum {count} bullets")
    lines.append("Do NOT produce fewer. You may add more. Never merge or drop bullets.")
    return "\n".join(lines)


# ── Diagnose counter output ───────────────────────────────────────────────────

print("=" * 70)
print("BULLET COUNTER OUTPUT (what gets injected into the prompt):")
print("=" * 70)
constraints = build_bullet_constraints(BASE_RESUME)
print(constraints)

counts = count_bullets_per_role(BASE_RESUME)
print("\nRaw counts dict:", json.dumps(counts, indent=2))

# ── Call the model ────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("\nNo OPENAI_API_KEY — skipping live model test.")
    exit(0)

from app import SYSTEM_PROMPT, build_bullet_constraints as bbc

role_label = "QA Automation Engineer"
sector_label = "Fintech startup"

user_message = f"""TARGET ROLE: {role_label}
TARGET SECTOR: {sector_label}

══════════════════════════════
{bbc(BASE_RESUME)}
══════════════════════════════

══════════════════════════════
BASE RESUME
══════════════════════════════
{BASE_RESUME}

══════════════════════════════
JOB DESCRIPTION
══════════════════════════════
{JOB_DESCRIPTION}

Please rewrite my resume following all the rules in your instructions. Tailor it specifically for the {role_label} role at a {sector_label} company based on the job description above.

REMINDER: Check the MANDATORY BULLET COUNTS above and verify your output meets or exceeds each minimum before finishing."""

print("\n" + "=" * 70)
print("CALLING MODEL (gpt-4o, non-streaming)...")
print("=" * 70)

client = OpenAI(api_key=api_key)
resp = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=8000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ],
    stream=False,
)

output = resp.choices[0].message.content
print("\nFULL OUTPUT:\n")
print(output)

# ── Count bullets in output ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("BULLET COUNT IN MODEL OUTPUT:")
print("=" * 70)
out_counts = count_bullets_per_role(output)
print(json.dumps(out_counts, indent=2))

# Identify shortfalls
print("\nSHORTFALL ANALYSIS (base resume counts vs output counts):")
base_counts = count_bullets_per_role(BASE_RESUME)
print(f"Base resume bullet counts: {base_counts}")
print(f"Output bullet counts:      {out_counts}")
