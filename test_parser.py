#!/usr/bin/env python3
import re

_BULLET_RE = re.compile(r'^\s*[-•·*]\s+')
_SKIP_SECTIONS = re.compile(
    r'^\s*(certifications?|education|qualifications?|key skills?|skills?|summary|profile)\s*$',
    re.IGNORECASE
)

def _clean(s):
    return re.sub(r'[\t ]+', ' ', s).strip()

def parse_roles_with_bullets(resume_text):
    lines = resume_text.splitlines()
    roles = []
    current_heading = None
    current_bullets = []
    in_work = False

    for line in lines:
        clean = _clean(line)
        if not clean:
            continue
        if _BULLET_RE.match(line):
            bullet_text = _BULLET_RE.sub('', line).strip()
            if current_heading and in_work:
                current_bullets.append(bullet_text)
        else:
            if current_heading and current_bullets:
                roles.append((current_heading, list(current_bullets)))
            elif current_heading and not current_bullets:
                pass
            if _SKIP_SECTIONS.match(clean):
                current_heading = None
                current_bullets = []
                continue
            if re.match(r'^\s*work\s+experience\s*$', clean, re.IGNORECASE):
                in_work = True
                current_heading = None
                current_bullets = []
                continue
            if in_work:
                if current_bullets:
                    current_heading = clean
                    current_bullets = []
                else:
                    if current_heading is None:
                        current_heading = clean
                    else:
                        if not re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|london|uk|india|present)\b', clean, re.IGNORECASE):
                            current_heading = clean
    if current_heading and current_bullets:
        roles.append((current_heading, current_bullets))
    return roles


BASE_RESUME = """Cloud Engineer (AWS) specialising in multi-account operations, infrastructure automation, and secure platform delivery. Experienced operating AWS environments supporting 250+ users across 7 AWS accounts, delivering repeatable baselines, data transfer/archival pipelines, and cost-optimised compute strategies. Strong hands-on skills in IaC (CloudFormation/Terraform), Linux, Python automation, networking fundamentals (VPC), and observability to improve reliability and operational response.
Key Skills
AWS: EC2, IAM, S3, VPC, CloudWatch, CloudTrail, Secrets Manager, IAM Identity Center, Route 53, KMS.
IaC: CloudFormation, Terraform, YAML, JSON.
Container: Docker, Kubernetes, ECS, EKS.
Automation: Python, Bash, PowerShell, REST APIs, Ansible.
CI/CD & Version Control: GitHub Actions, GitLab CI, Git, Jira, Confluence.
WORK EXPERIENCE
Konsistent Consulting (AWS Partner)                                             London, UK
Cloud Engineer                                                                           January 2025 - Present
* Operated 7 AWS accounts and customer facing services for 250+ users, owned access controls (IAM/Identity Center), compute (EC2), storage (S3) and networking (VPC) with CloudWatch telemetry and CloudTrail audit trails.
* Implemented an internal AI chat assistant for AWS Deadline Cloud by implementing an MCP server from scratch, enabling fast self-serve visibility into render farms and speeding up incident triage and capacity decisions during peak demand.
* Delivered a versioned multi-account baseline in 6 minutes using CloudFormation (with targeted Terraform guardrails), standardising logging, security posture, and account setup for repeatable delivery.
* Developed a real-time capacity and allocation visibility tool across regions/AZs to reduce insufficient capacity blockers by 40% and improve provisioning efficiency during peak demand.
* Eliminated 10k/month avoidable spend via Spot-first scaling, rightsizing, lifecycle policies, and resource governance, drove structured cost reviews and remediation using utilisation signals.
* Designed large-scale data transfer and archival workflows to S3 from on-prem storage with Python validation and deduplication, increased backup efficiency 70% and saved 30k/year through lifecycle tiering.
* Established CI-driven build and release pipelines (GitHub Actions) to build/test/publish versioned runtime packages; lowered environment failures 80% through dependency pinning and smoke test coverage.
* Standardised isolated execution for packages using Docker and Kubernetes (EKS) to run repeatable validation workloads and minimise drift across environments.
Milk VFX                                                                                    London, UK
Render Wrangler / Developer                                                         March 2024 - January 2025
* Streamlined deployment and configuration tasks for distributed compute workloads using Python/Bash, reducing manual operational overhead and improving repeatability.
* Hardened job submission and validation tooling with guardrails and configuration checks, increasing first-pass success rates and reducing support tickets (25% improvement in error resolution).
* Implemented operational dashboards and alerting using CloudWatch/Grafana and custom scripts, accelerating incident detection and reducing time-to-detect by 35% for common failure modes.
* Diagnosed and resolved infrastructure and application issues in production, improving utilisation (20%) and reducing downtime during delivery windows.
* Partnered with data analysts to extract workload level stats from platform reports and logs to guide optimisation and capacity planning.
Ghost VFX                                                                                   London, UK
Pipeline Developer                                                               February 2023 - February 2024
* Created automation tools that processed and validated 1M+ assets/work items with integrity checks, lowering errors by 15% through validation and guardrails.
* Released internal tools with structured versioning, documentation, testing, and regression checks, cutting deployment errors by 25%.
* Maintained GitLab CI pipelines with automated testing, packaging, and controlled deployments, increasing release reliability by 40%.
* Designed user-friendly tooling interfaces, improving workflow navigation for 15+ users and reducing task completion times by 40%.
Pixelloid Studios                                                                     Hyderabad, India
VFX Developer                                                                          July 2019 - July 2021
* Developed procedural tooling to generate 25+ variants of assets/workflows, reducing creation time by 30% through automation and reusable templates.
CERTIFICATIONS
* AWS Certified Cloud Practitioner
* AWS Solutions Architect Associate"""

roles = parse_roles_with_bullets(BASE_RESUME)
for heading, bullets in roles:
    heading_repr = repr(heading)
    print(f"ROLE: {heading_repr}  => {len(bullets)} bullets")
    for i, b in enumerate(bullets, 1):
        print(f"  {i}. {b[:100]}")
    print()

print("="*60)
print("FULL CONSTRAINT BLOCK (what gets injected):")
print("="*60)

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
print("\n".join(out))
