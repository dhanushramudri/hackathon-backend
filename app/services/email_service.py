import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("resourceiq.email")

JMAN_PURPLE = "#3411A3"
JMAN_PURPLE_DARK = "#19105B"

class EmailNotConfigured(Exception):
    pass

def _smtp_config() -> dict:
    host = os.environ.get("MAILTRAP_HOST", "")
    username = os.environ.get("MAILTRAP_USERNAME", "")
    password = os.environ.get("MAILTRAP_PASSWORD", "")
    if not (host and username and password):
        raise EmailNotConfigured("MAILTRAP_HOST/MAILTRAP_USERNAME/MAILTRAP_PASSWORD are not set in the environment.")
    return {
        "host": host,
        "port": int(os.environ.get("MAILTRAP_PORT", "2525")),
        "username": username,
        "password": password,
        "from_email": os.environ.get("MAILTRAP_FROM_EMAIL", "resourceiq@jmangroup.com"),
        "from_name": os.environ.get("MAILTRAP_FROM_NAME", "ResourceIQ"),
    }

def _wrap_jman_template(title: str, intro: str, sections_html: str) -> str:
    """Same visual language as JMAN's real email signature: a solid brand-purple
    banner, bold numbered sections, and a small confidentiality footer -- adapted
    for an automated system digest rather than a person's personal sign-off."""
    return f"""\
<!DOCTYPE html>
<html>
<body style="margin:0;padding:0;background:#F3F3F7;font-family:Arial,Helvetica,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#F3F3F7;padding:24px 0;">
    <tr><td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:10px;overflow:hidden;">
        <tr>
          <td style="background:{JMAN_PURPLE};padding:20px 28px;">
            <span style="color:#ffffff;font-size:18px;font-weight:bold;letter-spacing:0.3px;">JMAN Group</span>
            <span style="color:#C9BBF5;font-size:12px;display:block;margin-top:2px;">ResourceIQ — Automated Resourcing Digest</span>
          </td>
        </tr>
        <tr>
          <td style="padding:28px;color:#1F2937;font-size:13px;line-height:1.6;">
            <p style="margin:0 0 14px 0;">{intro}</p>
            {sections_html}
            <p style="margin:24px 0 0 0;">Regards,</p>
            <p style="margin:2px 0 0 0;font-weight:bold;color:{JMAN_PURPLE_DARK};">ResourceIQ | JMAN Group</p>
          </td>
        </tr>
        <tr>
          <td style="background:{JMAN_PURPLE_DARK};padding:14px 28px;">
            <span style="color:#C9BBF5;font-size:10px;">www.jmangroup.com</span>
          </td>
        </tr>
        <tr>
          <td style="padding:16px 28px;color:#9CA3AF;font-size:10px;line-height:1.5;font-style:italic;">
            This email, including any attachments, is generated automatically by ResourceIQ and may contain confidential
            resourcing information. If you are not the intended recipient, please notify the sender and delete this email.
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

# Mirrors frontend/lib/utils.ts's ROOT_CAUSE_LABEL so the email reads the same
# human-readable causes the Health page already shows, not raw snake_case codes.
ROOT_CAUSE_LABEL = {
    "overrunning": "Overrunning",
    "shadow_heavy": "Shadow-heavy",
    "high_churn": "High churn",
    "understaffed": "Understaffed",
    "overtime_risk": "Overtime risk",
    "effort_spike": "Effort spike",
    "wsr_deteriorating": "WSR getting worse",
    "wsr_critical": "WSR stuck at red/amber",
    "wsr_long_term_decline": "WSR fell and hasn't recovered",
}

def _root_cause_label(value: str) -> str:
    return ROOT_CAUSE_LABEL.get(value, value.replace("_", " "))

def _section(number: int, heading: str, items: list[str]) -> str:
    if not items:
        return f"""
            <p style="margin:0 0 4px 0;"><strong>{number}. {heading}</strong></p>
            <p style="margin:0 0 14px 0;color:#6B7280;">Nothing to flag here.</p>"""
    bullets = "".join(f'<li style="margin-bottom:4px;">{item}</li>' for item in items)
    return f"""
            <p style="margin:0 0 4px 0;"><strong>{number}. {heading}</strong></p>
            <ul style="margin:0 0 14px 0;padding-left:20px;">{bullets}</ul>"""

def render_digest_html(digest: dict, period_label: str) -> str:
    no_backfill_items = [
        f"<strong>{r['employee_id']}</strong> ({r['job_name'] or 'role unknown'}) on {r['project_id']} — "
        f"{r['leave_start_date']} → {r['leave_end_date']}, {r['allocation_by_percentage']}% allocated, no one free to cover."
        for r in digest["no_backfill_leaves"][:10]
    ]
    risk_items = []
    for p in digest["high_risk_projects"]:
        causes = ", ".join(_root_cause_label(c) for c in p["root_causes"]) or "elevated risk score"
        unbilled = p.get("monthly_unbilled_value_usd", 0)
        exposure = f" (${unbilled:,.0f}/mo unbilled exposure)" if unbilled > 0 else ""
        risk_items.append(f"<strong>{p['project_code']}</strong> — {causes}{exposure}")

    sections = _section(1, "Leave Coverage Gaps", no_backfill_items)
    if digest["no_backfill_count"] > len(no_backfill_items):
        sections += f'<p style="margin:-10px 0 14px 0;color:#6B7280;font-size:11px;">+{digest["no_backfill_count"] - len(no_backfill_items)} more — see the Leave page.</p>'
    sections += _section(2, "High-Risk Projects", risk_items)
    if digest["high_risk_total_count"] > len(risk_items):
        sections += f'<p style="margin:-10px 0 14px 0;color:#6B7280;font-size:11px;">+{digest["high_risk_total_count"] - len(risk_items)} more — see the Health page.</p>'

    intro = f"Dear Resource Manager,<br/>Here's what needs your attention {period_label}."
    return _wrap_jman_template(f"ResourceIQ Digest — {period_label}", intro, sections)

def send_email(to_email: str, subject: str, html_body: str) -> None:
    config = _smtp_config()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{config['from_name']} <{config['from_email']}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(config["host"], config["port"]) as server:
        server.starttls()
        server.login(config["username"], config["password"])
        server.sendmail(config["from_email"], [to_email], msg.as_string())
    logger.info("Digest email sent to %s", to_email)
