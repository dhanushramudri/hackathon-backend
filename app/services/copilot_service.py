import json
import logging
import re
from collections import Counter

import pandas as pd

from app.ai import llm
from app.ai.providers.base import QuotaExceededError
from app.core.adapter import get_adapter
from app.engines.coe_skill_engine import derive_skills_for_coes
from app.engines.designation_ladder import adjacent_designations
from app.engines.role_mix_engine import (
    get_role_mix,
    get_role_mix_by_category,
    get_role_mix_by_coes,
    list_coes,
    list_docx_categories,
)
from app.services.allocation_report_service import AllocationNotFound, get_allocation_report, get_allocation_timesheet
from app.services.demand_forecast_service import get_new_project_forecast
from app.services.employee_profile_service import (
    EmployeeNotFound,
    find_employees,
    get_employee_headcount_summary,
    get_employee_profile,
)
from app.services.free_pool_service import get_free_pool
from app.services.health_detail_service import ProjectNotFound, get_project_health_detail
from app.services.health_monitor_service import get_health_report
from app.services.leave_service import get_leave_impact
from app.services.pipeline_outlook_service import get_pipeline_outlook, get_pipeline_outlook_drilldown, get_six_month_outlook
from app.services.project_roster_service import get_project_info, get_project_roster
from app.services.rate_card_service import get_rate_card
from app.services.recommendation_service import (
    RowIndexOutOfRange,
    get_coverage_summary,
    get_recommendations,
    get_recommendations_for_pipeline_row,
    get_redeploy_matches_for_employee,
)
from app.services.revenue_service import get_revenue_trend
from app.services.semantic_match_service import get_semantic_match_suggestions
from app.services.timesheet_insights_service import get_employee_overtime_risk, get_project_effort_spikes

logger = logging.getLogger("resourceiq.copilot")

MAX_TOOL_TURNS = 6
MAX_HISTORY_TURNS = 8

SYSTEM_PROMPT = """You are the ResourceIQ copilot for JMAN's Resource Management Group (RMG).
You have full read access to every real engine in this app -- staffing recommendations,
free pool/redeployment, leave impact, employee profiles, project health/risk proof,
allocations/utilization, new-project demand forecasts, the flexible pipeline outlook,
role-mix/CoE/rate-card reference data, and timesheet-derived signals -- by calling the
available tools. Never guess or invent a number, an employee_id, or a project_code.
Today's reference date is the system date; if a question implies "now" or doesn't give a
date, use 2026-07-01 as a reasonable near-term default for staffing questions.

This system has NO ability to save, assign, approve, or change any real record -- it is
read-only end to end. Never phrase an answer as if you took an action ("I've assigned...",
"I've flagged..."); always phrase it as a finding or a recommendation for the RM to act on.

HOW TO CHAIN TOOLS (you are encouraged to call 2-3 tools per question when that gives a
materially more complete answer, not just the first one that looks plausible -- especially
for "why" or "what should I do" questions):
- If a question names a PERSON but you don't already have their employee_id from earlier
  in this conversation, call find_employees first, then the person-specific tool. Never
  invent an employee_id.
- If a question names a PROJECT but you don't already have its project_code, call
  get_health_report first (it lists every active project's code), then
  get_project_health_detail. get_health_report only covers ACTIVE projects -- if a
  project_code you already have (from allocations, roster, or the user directly) isn't
  in that list, it's likely PROPOSE/DEAL WON/CLOSED; use get_project_info (works for any
  status) and get_project_roster (who's ever been staffed on it, for churn/rotation
  questions) instead of treating it as not found.
- For "who's logged hours on this allocation" / "prove the hours" / "are there missing
  days" questions about one specific person+project, call get_allocation_timesheet with
  the employee_id and project_id (from get_employee_profile's allocations list or
  get_allocation_report -- never invent either).
- For revenue/revenue-trend/revenue-leakage questions, call get_revenue_trend.
- If asked "which deals/people specifically" about a number from get_pipeline_outlook,
  call get_pipeline_outlook_drilldown next using the exact month/value label from that
  prior result -- never invent a label.
- For a free/available person, call get_redeploy_matches_for_employee after get_free_pool
  or find_employees to show what open pipeline work they could actually fill.
- For pipeline-specific staffing questions ("what does <client>'s deal need", "who should
  staff the <role> request"), call list_pipeline_demand first to find the right row_index,
  then get_recommendations_for_pipeline_row. For portfolio-wide coverage questions, call
  get_recommendations_coverage_summary. For a free-text skillset not tied to a pipeline
  row, use get_recommendations directly. Only call get_semantic_match_suggestions when the
  user explicitly asks for a deeper second-opinion look at one specific row_index that
  already came back as a hire signal/weak match -- it is expensive, never call it
  speculatively, and never more than once per question.
- ALWAYS include the literal employee_id and/or project_code in your summary text when
  discussing one specific person or project, even if a table is also shown -- you will
  only see your own prior text (not raw tool data) on the next turn of this conversation,
  so a follow-up question about "that person"/"that project" can only be resolved if the
  id was actually written out in plain text before.
- If a tool returns no good candidates, say so plainly and call it a hire signal rather
  than softening it. Cite counts and percentages exactly as given in the tool result
  (e.g. use hire_signal_pct as-is) -- never compute your own derived percentage from a
  different denominator, even if the arithmetic looks simple; a percentage in your summary
  must match the structured table/stats shown alongside it exactly. Leave data in this
  system is SYNTHETIC (no real leave/absence dataset exists in the source files) -- say so
  plainly whenever you surface it. The Rate Card is ILLUSTRATIVE (no real cost data exists
  anywhere) -- say so plainly whenever you cite a dollar figure from it.

You have at most a few tool-call turns for one question -- use them deliberately, don't
call more tools than the question actually needs.

When you give your FINAL answer (no more tool calls needed), respond with ONLY a JSON
object, no other text: {"summary": "<2-4 concrete sentences, naming employee IDs/project
codes/counts>", "format": "table"|"stats"|"text"}. Choose "table" when the answer is
naturally a ranked or listed set of items (candidates, pipeline rows, projects,
allocations, free-pool people). Choose "stats" when the answer is a small set of counts/
percentages/totals (coverage summary, outlook warnings, shortfall figures, a single
project's or employee's headline numbers). Choose "text" for a narrative explanation with
no natural list or counts. Do not include markdown, code fences, or any text outside the
JSON object."""

TOOLS = [
    {
        "name": "get_recommendations",
        "description": "Rank internal employees against a required skillset for a given start date. Returns eligible/trainable/gap-bucketed candidates and a hire-vs-redeploy flag. Use list_pipeline_demand + get_recommendations_for_pipeline_row instead when the question is about a specific pipeline deal.",
        "parameters": {
            "type": "object",
            "properties": {
                "skillset_text": {"type": "string", "description": "Comma-separated required skills, e.g. 'Python, ETL pipelines, AWS'"},
                "likely_start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "requested_pct_raw": {"type": "string", "description": "Requested allocation percent, e.g. '100' or '50'"},
            },
            "required": ["skillset_text", "likely_start_date"],
        },
    },
    {
        "name": "list_pipeline_demand",
        "description": "Search/browse the pipeline demand list (one row per requested role per deal) by client name, requested role/grade, skillset, or solution keyword. Returns row_index values needed by get_recommendations_for_pipeline_row. Omit query to list the first 50 rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Case-insensitive keyword to filter by client, requested role/grade, skillset, or solution name"},
            },
        },
    },
    {
        "name": "get_recommendations_for_pipeline_row",
        "description": "Ranked candidate recommendations for one specific pipeline demand row (found via list_pipeline_demand): real deal context (client, EM, SOW status, requested grade), ranked candidates with skill/competency/availability scores, and a plain-English explanation per candidate.",
        "parameters": {
            "type": "object",
            "properties": {
                "row_index": {"type": "integer", "description": "The row_index from list_pipeline_demand"},
            },
            "required": ["row_index"],
        },
    },
    {
        "name": "get_recommendations_coverage_summary",
        "description": "Portfolio-wide rollup across every pipeline demand row: how many role-requests are ready to redeploy internally, need upskilling, need an external hire, or have no skillset specified yet. Use for 'how much of the pipeline is covered' style questions.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_health_report",
        "description": "Returns every active project's derived risk score, risk band (high/medium/low), root-cause tags (overrunning, shadow_heavy, high_churn), and real WSR signal where available.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_allocation_report",
        "description": "Returns the current-state allocation table: every active allocation with FTE%, dates, billing status, utilization band (over_allocated/normal/under_utilized), and whether it's ending within 30 days.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_new_project_forecast",
        "description": "Given a list of hypothetical new projects (standard project category name, count), computes headcount need via the role-mix engine and checks it against the current redeployment pool, returning shortfall/hire signals by designation with the specific candidate employees/projects that could free up capacity.",
        "parameters": {
            "type": "object",
            "properties": {
                "specs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": [
                                    "D&D Tactical Build", "Build Phase - Tactical Build", "Build Phase - Enterprise Platform Build",
                                    "Build Phase - Data Platform Build", "Data Science Projects", "AI Projects",
                                    "MS projects", "Full stack Projects", "Value creation Projects",
                                ],
                            },
                            "count": {"type": "integer"},
                        },
                        "required": ["category", "count"],
                    },
                }
            },
            "required": ["specs"],
        },
    },
    {
        "name": "find_employees",
        "description": "Look up real employee_id values by partial name/role/department/location keyword. Call this FIRST whenever a question names a person, role, or department but you don't already have their employee_id from earlier in this conversation. Never invent an employee_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Case-insensitive keyword: part of a job title, department, location, or employee_id"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_employee_profile",
        "description": "Full single-employee drill-down: skills (observed/imputed), competencies, complete allocation history, current allocations with hours, overtime risk, recent daily hours, leave records, and computed signals (over_allocated/under_utilized/sustained_overtime/possible_unplanned_absence). Requires a real employee_id -- use find_employees first if you don't have one.",
        "parameters": {
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"],
        },
    },
    {
        "name": "get_free_pool",
        "description": "Everyone currently with spare capacity or no allocation at all (fully_free/under_utilized/ending_soon), with idle capacity %, idle $/month value, primary CoE, and days_free. Use for 'who's available right now' style questions, distinct from a skill-matched search.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_redeploy_matches_for_employee",
        "description": "For one specific free/available employee, the real open (not yet fully Resourced) pipeline deals their own skills overlap with -- the reverse direction of get_recommendations. Requires a real employee_id (use find_employees or get_free_pool first).",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "top_n": {"type": "integer", "description": "Max matches to return, default 5"},
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "get_leave_impact",
        "description": "Upcoming/current leave records cross-referenced against active allocations, with same-designation backfill candidates per affected project. Leave data in this system is SYNTHETIC/illustrative -- always say so when surfacing it.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_employee_headcount_summary",
        "description": "Real current headcount: currently active vs. already-departed vs. in-notice-period. Use for 'how many employees do we have' style questions -- never use a raw row count from any other tool for this.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_project_health_detail",
        "description": "The full real-rows proof behind one project's fired risk root causes (overrun, shadow-heavy, high churn, understaffed, overtime risk, effort spike, and the 3 WSR signals). Requires a real project_code -- call get_health_report first if you don't already have one from this conversation.",
        "parameters": {
            "type": "object",
            "properties": {"project_code": {"type": "string"}},
            "required": ["project_code"],
        },
    },
    {
        "name": "get_pipeline_outlook",
        "description": "Confirmed vs unconfirmed pipeline demand and projected supply over a flexible date range/horizon/granularity, with shortfall math (confirmed only), skill-area breakdown, and cluster account scorecards. Use start_date/horizon_months/granularity to answer 'next year', 'by week', 'starting next quarter' style questions -- not limited to a fixed 6 months.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY-MM-DD, defaults to tomorrow"},
                "horizon_months": {"type": "integer", "description": "Defaults to 6, max 36"},
                "granularity": {"type": "string", "enum": ["month", "week"]},
            },
        },
    },
    {
        "name": "get_pipeline_outlook_drilldown",
        "description": "The literal real deals or employees behind one number from get_pipeline_outlook. Call this after get_pipeline_outlook when the user asks 'which deals/which employees specifically' about one month/role/cluster/skill_area.",
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": ["confirmed_demand", "unconfirmed_demand", "supply", "role", "skill_area", "cluster", "solution"]},
                "value": {"type": "string", "description": "Required for role/skill_area/cluster/solution -- the exact label as shown in get_pipeline_outlook's output"},
                "month": {"type": "string", "description": "The exact month/week label as shown in get_pipeline_outlook's output"},
                "is_confirmed": {"type": "boolean", "description": "Only matters for dimension=role, defaults to true"},
            },
            "required": ["dimension"],
        },
    },
    {
        "name": "get_semantic_match_suggestions",
        "description": "On-demand AI-assisted broader-phrasing skill match for ONE pipeline row that already came back as a hire signal/gap. Only call this when the user explicitly asks for a deeper/second-opinion look at a specific row_index -- it is expensive, never call it speculatively or for more than one row per question.",
        "parameters": {
            "type": "object",
            "properties": {"row_index": {"type": "integer"}},
            "required": ["row_index"],
        },
    },
    {
        "name": "get_role_mix",
        "description": "Typical role-mix (designation -> headcount/FTE) for a project type, either by one of the 9 fixed standard project categories, or by a custom list of CoEs + optional type_of_project. Use for 'what roles does a typical X project need' questions. Use list_role_mix_reference first if you need the valid category/CoE names.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "One of the 9 fixed standard category names -- use this OR coes, not both"},
                "coes": {"type": "array", "items": {"type": "string"}, "description": "Canonical CoE names, e.g. ['AI & ML', 'Data Engineering'] -- use this OR category"},
                "type_of_project": {"type": "string", "description": "Optional narrowing, only used with coes"},
            },
        },
    },
    {
        "name": "list_role_mix_reference",
        "description": "Reference lookup: the 5 canonical Centers of Excellence with real historical sample sizes, and the 9 fixed standard project categories with their resolved role-mix source. Call this if you need valid values to pass to get_role_mix or get_new_project_forecast.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_rate_card",
        "description": "Illustrative hourly billing rate for one or more job titles/designations.",
        "parameters": {
            "type": "object",
            "properties": {"job_names": {"type": "array", "items": {"type": "string"}}},
            "required": ["job_names"],
        },
    },
    {
        "name": "get_adjacent_designations",
        "description": "Seniority-ladder neighbors (one level up/down) for a designation -- use to explain why a near-miss candidate one level off from the requested grade might still be a reasonable fit.",
        "parameters": {
            "type": "object",
            "properties": {
                "designation": {"type": "string"},
                "max_levels": {"type": "integer", "description": "Defaults to 1"},
            },
            "required": ["designation"],
        },
    },
    {
        "name": "get_employee_overtime_risk",
        "description": "Portfolio-wide: every employee currently flagged for sustained overtime risk, with recent overtime day counts and peak daily hours.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_project_effort_spikes",
        "description": "Portfolio-wide: every project whose latest week's logged hours spiked vs. its own recent baseline -- an overrun early-warning signal distinct from get_health_report's already-fired root causes.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_coe_skills",
        "description": "Typical real skills observed for one or more Centers of Excellence (low-to-medium confidence, explicitly labeled). Use for 'what skills should I look for if hiring/redeploying into CoE X' questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "coes": {"type": "array", "items": {"type": "string"}},
                "top_n": {"type": "integer", "description": "Defaults to 12"},
            },
            "required": ["coes"],
        },
    },
    {
        "name": "get_allocation_timesheet",
        "description": "Day-by-day timesheet proof behind one specific employee+project allocation: every calendar day in the window with hours logged, expected hours, and any day with NO timesheet entry explicitly flagged as missing (distinct from a real logged zero). Requires a real employee_id and project_id, e.g. from get_employee_profile's allocations list or get_allocation_report.",
        "parameters": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "project_id": {"type": "string"},
            },
            "required": ["employee_id", "project_id"],
        },
    },
    {
        "name": "get_project_roster",
        "description": "Every employee ever allocated to one project (full history, not just current), with allocation %, start/end dates, and active flag -- answers 'who's worked on this project' and churn/rotation questions. Works for any project_code regardless of status.",
        "parameters": {
            "type": "object",
            "properties": {"project_code": {"type": "string"}},
            "required": ["project_code"],
        },
    },
    {
        "name": "get_project_info",
        "description": "Basic project lookup (status, type, CoE, start/end dates) for ANY project_code regardless of status -- use this when a project isn't in get_health_report's ACTIVE-only list (e.g. still PROPOSE/DEAL WON, or already CLOSED).",
        "parameters": {
            "type": "object",
            "properties": {"project_code": {"type": "string"}},
            "required": ["project_code"],
        },
    },
    {
        "name": "get_revenue_trend",
        "description": "Monthly revenue trend from the pipeline workbook's revenue sheet. Use for 'what's our revenue trend / revenue leakage' style questions.",
        "parameters": {"type": "object", "properties": {}},
    },
]

def _list_pipeline_demand(query: str | None = None) -> list[dict]:
    pipeline = get_adapter().get_pipeline_forecast().reset_index().rename(columns={"index": "row_index"})
    if query:
        q = query.lower()
        text_cols = ["client", "resources_requested", "skillset", "solution"]
        mask = pd.Series(False, index=pipeline.index)
        for col in text_cols:
            mask |= pipeline[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
        pipeline = pipeline[mask]

    cols = ["row_index", "cluster", "client", "client_priority", "solution", "resources_requested", "skillset", "likely_start_date", "sow_signed"]
    out = pipeline[cols].head(50).copy()
    out["row_index"] = out["row_index"].astype(int)
    out["cluster"] = out["cluster"].apply(lambda v: int(v) if pd.notna(v) else None)
    out["likely_start_date"] = out["likely_start_date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")

def _dispatch(name: str, args: dict):
    if name == "get_recommendations":
        return get_recommendations(
            skillset_text=args.get("skillset_text", ""),
            likely_start_date=args.get("likely_start_date", "2026-07-01"),
            requested_pct_raw=args.get("requested_pct_raw", "100"),
        )
    if name == "list_pipeline_demand":
        return _list_pipeline_demand(args.get("query"))
    if name == "get_recommendations_for_pipeline_row":
        try:
            return get_recommendations_for_pipeline_row(int(args.get("row_index", -1)))
        except (RowIndexOutOfRange, TypeError, ValueError) as exc:
            return {"error": str(exc)}
    if name == "get_recommendations_coverage_summary":
        return get_coverage_summary()
    if name == "get_health_report":
        return get_health_report()
    if name == "get_allocation_report":
        return get_allocation_report()
    if name == "get_new_project_forecast":
        return get_new_project_forecast(args.get("specs", []))
    if name == "find_employees":
        return find_employees(args.get("query", ""))
    if name == "get_employee_profile":
        try:
            return get_employee_profile(args.get("employee_id", ""))
        except EmployeeNotFound as exc:
            return {"error": str(exc)}
    if name == "get_free_pool":
        return get_free_pool()
    if name == "get_redeploy_matches_for_employee":
        return get_redeploy_matches_for_employee(args.get("employee_id", ""), int(args.get("top_n", 5)))
    if name == "get_leave_impact":
        return get_leave_impact()
    if name == "get_employee_headcount_summary":
        return get_employee_headcount_summary()
    if name == "get_project_health_detail":
        try:
            return get_project_health_detail(args.get("project_code", ""))
        except ProjectNotFound as exc:
            return {"error": str(exc)}
    if name == "get_pipeline_outlook":
        return get_pipeline_outlook(
            start_date=args.get("start_date"),
            horizon_months=int(args.get("horizon_months", 6)),
            granularity=args.get("granularity", "month"),
        )
    if name == "get_pipeline_outlook_drilldown":
        return get_pipeline_outlook_drilldown(
            dimension=args.get("dimension", ""),
            value=args.get("value"),
            month=args.get("month"),
            is_confirmed=bool(args.get("is_confirmed", True)),
        )
    if name == "get_semantic_match_suggestions":
        try:
            return get_semantic_match_suggestions(int(args.get("row_index", -1)))
        except (RowIndexOutOfRange, TypeError, ValueError) as exc:
            return {"error": str(exc)}
    if name == "get_role_mix":
        if args.get("category"):
            return get_role_mix_by_category(args["category"])
        return get_role_mix_by_coes(args.get("coes", []), args.get("type_of_project"))
    if name == "list_role_mix_reference":
        return {"coes": list_coes(), "project_categories": list_docx_categories()}
    if name == "get_rate_card":
        return get_rate_card(args.get("job_names", []))
    if name == "get_adjacent_designations":
        pairs = adjacent_designations(args.get("designation", ""), int(args.get("max_levels", 1)))
        return [{"designation": d, "level_offset": o} for d, o in pairs]
    if name == "get_employee_overtime_risk":
        return get_employee_overtime_risk()
    if name == "get_project_effort_spikes":
        return get_project_effort_spikes()
    if name == "get_coe_skills":
        return derive_skills_for_coes(args.get("coes", []), int(args.get("top_n", 12)))
    if name == "get_allocation_timesheet":
        try:
            return get_allocation_timesheet(args.get("employee_id", ""), args.get("project_id", ""))
        except AllocationNotFound as exc:
            return {"error": str(exc)}
    if name == "get_project_roster":
        return get_project_roster(args.get("project_code", ""))
    if name == "get_project_info":
        info = get_project_info(args.get("project_code", ""))
        return info if info is not None else {"error": f"project {args.get('project_code')} not found"}
    if name == "get_revenue_trend":
        return get_revenue_trend()
    return {"error": f"unknown tool {name}"}

def _truncate_for_llm(name: str, result):
    if name == "get_health_report" and isinstance(result, list):
        return sorted(result, key=lambda r: -r["risk_score"])[:15]
    if name == "get_allocation_report" and isinstance(result, list):
        return result[:20]
    if name in ("get_recommendations", "get_recommendations_for_pipeline_row") and isinstance(result, dict):
        return {**result, "candidates": result.get("candidates", [])[:8]}
    if name == "list_pipeline_demand" and isinstance(result, list):
        return result[:30]
    if name == "get_recommendations_coverage_summary" and isinstance(result, dict):
        return {k: v for k, v in result.items() if k != "rows"}
    if name == "get_project_health_detail" and isinstance(result, dict):
        trimmed = {k: v for k, v in result.items() if k != "allocations_roster"}
        for section in ("overrun", "shadow_heavy", "high_churn", "overtime_risk", "effort_spike", "wsr"):
            if isinstance(trimmed.get(section), dict):
                trimmed[section] = {
                    k: v for k, v in trimmed[section].items()
                    if k not in ("qualifying_allocations", "roster_timeline", "employees", "weekly_hours", "reports")
                }
        return trimmed
    if name == "get_employee_profile" and isinstance(result, dict):
        return {
            **result,
            "skills": result.get("skills", [])[:10],
            "competencies": result.get("competencies", [])[:10],
            "allocations": result.get("allocations", [])[:10],
            "daily_hours_recent": result.get("daily_hours_recent", [])[:5],
        }
    if name == "get_free_pool" and isinstance(result, list):
        return result[:25]
    if name == "get_leave_impact" and isinstance(result, list):
        return result[:20]
    if name == "get_pipeline_outlook" and isinstance(result, dict):
        return {
            k: v for k, v in result.items()
            if k not in ("role_demand_by_month", "skill_area_demand_by_month", "project_mix_by_cluster_by_month", "project_mix_by_solution_by_month", "cluster_scorecards")
        }
    if name == "get_pipeline_outlook_drilldown" and isinstance(result, dict):
        return {
            **result,
            "deals": result.get("deals", [])[:15],
            "supply_employees": result.get("supply_employees", [])[:15],
            "designation_roster": result.get("designation_roster", [])[:15],
        }
    if name == "get_redeploy_matches_for_employee" and isinstance(result, list):
        return result[:5]
    if name == "get_employee_overtime_risk" and isinstance(result, dict):
        at_risk = {k: v for k, v in result.items() if v.get("is_sustained_overtime")}
        return dict(list(at_risk.items())[:20])
    if name == "get_project_effort_spikes" and isinstance(result, dict):
        spikes = {k: v for k, v in result.items() if v.get("is_effort_spike")}
        return dict(list(spikes.items())[:20])
    if name == "get_allocation_timesheet" and isinstance(result, dict):
        return {**result, "daily_hours": result.get("daily_hours", [])[-30:]}
    if name == "get_project_roster" and isinstance(result, dict):
        return {**result, "roster": result.get("roster", [])[:20]}
    return result

_BUCKET_LABEL = {"eligible": "Redeploy", "trainable": "Needs training", "gap": "Hire signal", "not_assessed": "Not assessed"}

def _candidates_table(data) -> dict | None:
    candidates = (data or {}).get("candidates") if isinstance(data, dict) else None
    if not candidates:
        return None
    columns = ["Employee", "Role", "Signal", "Skill match", "Competency", "Available %"]
    rows = [
        [
            c["employee_id"],
            c.get("job_name") or "-",
            _BUCKET_LABEL.get(c.get("bucket"), c.get("bucket", "-")),
            f"{c.get('skill_score', 0):.2f}",
            f"{c.get('competency_score', 0):.2f}",
            f"{c.get('available_pct', 0):.0f}%",
        ]
        for c in candidates[:10]
    ]
    return {"columns": columns, "rows": rows}

def _pipeline_demand_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    columns = ["#", "Client", "Role requested", "Likely start", "SOW signed"]
    rows = [
        [r.get("row_index"), r.get("client") or "Unnamed", r.get("resources_requested") or "-", r.get("likely_start_date") or "-", r.get("sow_signed") or "unconfirmed"]
        for r in data[:15]
    ]
    return {"columns": columns, "rows": rows}

def _health_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    ranked = sorted(data, key=lambda r: -r.get("risk_score", 0))[:10]
    columns = ["Project", "Risk", "Root causes", "Unbilled $/mo"]
    rows = [
        [p.get("project_code"), p.get("risk_band"), ", ".join(p.get("root_causes", [])) or "-", f"${p.get('monthly_unbilled_value_usd', 0):,.0f}"]
        for p in ranked
    ]
    return {"columns": columns, "rows": rows}

def _allocation_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    columns = ["Employee", "Project", "Allocation %", "Status", "Ending soon"]
    rows = [
        [r.get("employee_id"), r.get("project_id"), f"{r.get('allocation_by_percentage', 0):.0f}%", r.get("utilization_band"), "Yes" if r.get("ending_soon") else "No"]
        for r in data[:10]
    ]
    return {"columns": columns, "rows": rows}

def _forecast_table(data) -> dict | None:
    breakdown = (data or {}).get("breakdown") if isinstance(data, dict) else None
    if not breakdown:
        return None
    columns = ["Designation", "Needed", "Redeploy available", "Shortfall", "Shortfall $/mo"]
    rows = [
        [b.get("designation"), b.get("needed_headcount"), b.get("available_for_redeploy"), b.get("shortfall"), f"${b.get('shortfall_value_usd', 0):,.0f}"]
        for b in breakdown
    ]
    return {"columns": columns, "rows": rows}

def _outlook_table(data) -> dict | None:
    months = (data or {}).get("months") if isinstance(data, dict) else None
    if not months:
        return None
    columns = ["Month", "Confirmed demand", "Unconfirmed demand", "Projected supply", "Early warning"]
    rows = [
        [m.get("month"), m.get("confirmed_demand_count"), m.get("unconfirmed_demand_count"), m.get("projected_supply_count"), "Yes" if m.get("early_warning") else "No"]
        for m in months
    ]
    return {"columns": columns, "rows": rows}

def _free_pool_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    columns = ["Employee", "Designation", "CoE", "Status", "Idle %", "Idle $/mo"]
    rows = [
        [
            c.get("employee_id"), c.get("job_name") or "-", c.get("primary_coe") or "not determined",
            c.get("reason"), f"{c.get('idle_capacity_pct', 0):.0f}%",
            f"${c['idle_value_usd_per_month']:,.0f}" if c.get("idle_value_usd_per_month") is not None else "non-billable",
        ]
        for c in data[:10]
    ]
    return {"columns": columns, "rows": rows}

def _leave_impact_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    columns = ["Employee", "Project", "Leave type", "Starts", "Ends", "Backfill available"]
    rows = [
        [
            r.get("employee_id"), r.get("project_id") or "-", r.get("leave_type"),
            r.get("leave_start_date") or "-", r.get("leave_end_date") or "-",
            "Yes" if r.get("backfill_available") else "No",
        ]
        for r in data[:10]
    ]
    return {"columns": columns, "rows": rows}

def _redeploy_matches_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    columns = ["Client", "Role requested", "Likely start", "Skill match"]
    rows = [
        [m.get("client") or "Unnamed", m.get("resources_requested") or "-", m.get("likely_start_date") or "-", f"{m.get('skill_score', 0) * 100:.0f}%"]
        for m in data[:10]
    ]
    return {"columns": columns, "rows": rows}

def _rate_card_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    columns = ["Job title", "Hourly rate ($)"]
    rows = [[r.get("job_name"), f"${r['hourly_rate_usd']:.0f}" if r.get("hourly_rate_usd") is not None else "non-billable"] for r in data[:15]]
    return {"columns": columns, "rows": rows}

def _overtime_risk_table(data) -> dict | None:
    if not isinstance(data, dict) or not data:
        return None
    columns = ["Employee", "Overtime days (14d)", "Max daily hours"]
    rows = [[emp_id, r.get("overtime_days_recent"), r.get("max_daily_hours_recent")] for emp_id, r in list(data.items())[:10]]
    return {"columns": columns, "rows": rows}

def _effort_spike_table(data) -> dict | None:
    if not isinstance(data, dict) or not data:
        return None
    columns = ["Project", "Latest week hrs", "Baseline avg hrs"]
    rows = [[proj_id, r.get("latest_week_hours"), r.get("baseline_avg_weekly_hours")] for proj_id, r in list(data.items())[:10]]
    return {"columns": columns, "rows": rows}

def _role_mix_table(data) -> dict | None:
    roles = (data or {}).get("roles") if isinstance(data, dict) else None
    if not roles:
        return None
    columns = ["Designation", "Headcount", "Typical %", "Prevalence %"]
    rows = [[r.get("designation"), r.get("headcount"), f"{r.get('typical_pct', 0):.0f}%", f"{r.get('prevalence_pct')}%" if r.get("prevalence_pct") is not None else "-"] for r in roles]
    return {"columns": columns, "rows": rows}

def _outlook_drilldown_table(data) -> dict | None:
    if not isinstance(data, dict):
        return None
    deals = data.get("deals") or []
    if deals:
        columns = ["Client", "Role", "Likely start", "Confirmed", "Value $"]
        rows = [
            [d.get("client") or "Unnamed", d.get("role_label") or "-", d.get("likely_start_date") or "-",
             "Yes" if d.get("is_confirmed") else "No", f"${d['value_usd']:,.0f}" if d.get("value_usd") is not None else "-"]
            for d in deals[:10]
        ]
        return {"columns": columns, "rows": rows}
    employees = data.get("supply_employees") or []
    if employees:
        columns = ["Employee", "Project", "Ends"]
        rows = [[e.get("employee_id"), e.get("project_id") or "-", e.get("allocated_end_date") or "-"] for e in employees[:10]]
        return {"columns": columns, "rows": rows}
    return None

def _timesheet_table(data) -> dict | None:
    daily = (data or {}).get("daily_hours") if isinstance(data, dict) else None
    if not daily:
        return None
    columns = ["Date", "Hours logged", "Expected hours", "Status"]
    rows = [
        [
            d.get("date"),
            f"{d['hours']:.2f}" if d.get("hours") is not None else "-",
            d.get("expected_hours"),
            "Missing" if d.get("is_missing") else (f"{d['utilization_pct']:.0f}%" if d.get("utilization_pct") is not None else "-"),
        ]
        for d in daily[-20:]
    ]
    return {"columns": columns, "rows": rows}

def _roster_table(data) -> dict | None:
    roster = (data or {}).get("roster") if isinstance(data, dict) else None
    if not roster:
        return None
    columns = ["Employee", "Role", "Allocation %", "Start", "End", "Active"]
    rows = [
        [
            r.get("employee_id"), r.get("job_name") or "-", f"{r.get('allocation_by_percentage', 0):.0f}%",
            r.get("allocated_start_date") or "-", r.get("allocated_end_date") or "-",
            "Yes" if r.get("is_allocation_active") else "No",
        ]
        for r in roster[:15]
    ]
    return {"columns": columns, "rows": rows}

def _revenue_table(data) -> dict | None:
    if not isinstance(data, list) or not data:
        return None
    columns = ["Month", "Revenue ($)"]
    rows = [[r.get("month"), f"${r.get('value', 0):,.0f}"] for r in data]
    return {"columns": columns, "rows": rows}

_TABLE_BUILDERS = {
    "get_recommendations": _candidates_table,
    "get_recommendations_for_pipeline_row": _candidates_table,
    "list_pipeline_demand": _pipeline_demand_table,
    "get_health_report": _health_table,
    "get_allocation_report": _allocation_table,
    "get_new_project_forecast": _forecast_table,
    "get_pipeline_outlook": _outlook_table,
    "get_free_pool": _free_pool_table,
    "get_leave_impact": _leave_impact_table,
    "get_redeploy_matches_for_employee": _redeploy_matches_table,
    "get_rate_card": _rate_card_table,
    "get_employee_overtime_risk": _overtime_risk_table,
    "get_project_effort_spikes": _effort_spike_table,
    "get_role_mix": _role_mix_table,
    "get_pipeline_outlook_drilldown": _outlook_drilldown_table,
    "get_allocation_timesheet": _timesheet_table,
    "get_project_roster": _roster_table,
    "get_revenue_trend": _revenue_table,
}

def _build_table(tool_name: str | None, data) -> dict | None:
    builder = _TABLE_BUILDERS.get(tool_name)
    return builder(data) if builder else None

def _coverage_stats(data) -> list[dict] | None:
    if not isinstance(data, dict) or "total_demand_rows" not in data:
        return None
    return [
        {"label": "Ready to redeploy", "value": str(data.get("redeploy_ready_count", 0))},
        {"label": "Need upskilling", "value": str(data.get("redeploy_with_training_count", 0))},
        {"label": "Need external hire", "value": f"{data.get('hire_signal_count', 0)} ({data.get('hire_signal_pct', 0)}%)"},
        {"label": "No skillset specified", "value": str(data.get("no_skillset_specified_count", 0))},
    ]

def _outlook_stats(data) -> list[dict] | None:
    months = (data or {}).get("months") if isinstance(data, dict) else None
    if not months:
        return None
    warnings = sum(1 for m in months if m.get("early_warning"))
    confirmed = sum(m.get("confirmed_demand_count", 0) for m in months)
    unconfirmed = sum(m.get("unconfirmed_demand_count", 0) for m in months)
    return [
        {"label": "Months with shortfall warning", "value": f"{warnings} / {len(months)}"},
        {"label": "Total confirmed demand", "value": str(confirmed)},
        {"label": "Total unconfirmed demand", "value": str(unconfirmed)},
    ]

def _forecast_stats(data) -> list[dict] | None:
    if not isinstance(data, dict) or "total_shortfall_headcount" not in data:
        return None
    return [
        {"label": "Total shortfall (heads)", "value": str(data.get("total_shortfall_headcount", 0))},
        {"label": "Total shortfall value", "value": f"${data.get('total_shortfall_value_usd', 0):,.0f}/mo"},
    ]

def _health_stats(data) -> list[dict] | None:
    if not isinstance(data, list) or not data:
        return None
    counts = Counter(p.get("risk_band") for p in data)
    return [
        {"label": "High risk", "value": str(counts.get("high", 0))},
        {"label": "Medium risk", "value": str(counts.get("medium", 0))},
        {"label": "Low risk", "value": str(counts.get("low", 0))},
    ]

def _allocation_stats(data) -> list[dict] | None:
    if not isinstance(data, list) or not data:
        return None
    over = sum(1 for r in data if r.get("utilization_band") == "over_allocated")
    ending = sum(1 for r in data if r.get("ending_soon"))
    under = sum(1 for r in data if r.get("utilization_band") == "under_utilized")
    return [
        {"label": "Over-allocated", "value": str(over)},
        {"label": "Ending within 30 days", "value": str(ending)},
        {"label": "Under-utilized", "value": str(under)},
    ]

def _headcount_stats(data) -> list[dict] | None:
    if not isinstance(data, dict) or "currently_active" not in data:
        return None
    return [
        {"label": "Currently active", "value": str(data.get("currently_active", 0))},
        {"label": "Already departed", "value": str(data.get("already_departed", 0))},
        {"label": "In notice period", "value": str(data.get("in_notice_period", 0))},
    ]

def _employee_profile_stats(data) -> list[dict] | None:
    if not isinstance(data, dict) or "signals" not in data:
        return None
    s = data["signals"]
    return [
        {"label": "Designation", "value": data.get("job_name") or "-"},
        {"label": "Current allocation %", "value": f"{data.get('employee_total_allocation_pct')}%" if data.get("employee_total_allocation_pct") is not None else "0%"},
        {"label": "Over-allocated?", "value": "Yes" if s.get("over_allocated") else "No"},
        {"label": "Sustained overtime?", "value": "Yes" if s.get("sustained_overtime") else "No"},
        {"label": "Possible unplanned absence?", "value": "Yes" if s.get("possible_unplanned_absence") else "No"},
    ]

def _health_detail_stats(data) -> list[dict] | None:
    if not isinstance(data, dict) or "risk_band" not in data:
        return None
    return [
        {"label": "Risk band", "value": str(data.get("risk_band"))},
        {"label": "Risk score", "value": str(data.get("risk_score"))},
        {"label": "Root causes fired", "value": ", ".join(data.get("root_causes", [])) or "none"},
    ]

def _project_info_stats(data) -> list[dict] | None:
    if not isinstance(data, dict) or "project_status" not in data:
        return None
    return [
        {"label": "Status", "value": data.get("project_status") or "-"},
        {"label": "Type", "value": data.get("type_of_project") or "-"},
        {"label": "Tech CoE", "value": data.get("tech_coe") or "-"},
        {"label": "Start", "value": data.get("project_start_date") or "-"},
        {"label": "End", "value": data.get("project_end_date") or "-"},
    ]

_STATS_BUILDERS = {
    "get_recommendations_coverage_summary": _coverage_stats,
    "get_pipeline_outlook": _outlook_stats,
    "get_new_project_forecast": _forecast_stats,
    "get_health_report": _health_stats,
    "get_allocation_report": _allocation_stats,
    "get_employee_headcount_summary": _headcount_stats,
    "get_employee_profile": _employee_profile_stats,
    "get_project_health_detail": _health_detail_stats,
    "get_project_info": _project_info_stats,
}

def _build_stats(tool_name: str | None, data) -> list[dict] | None:
    builder = _STATS_BUILDERS.get(tool_name)
    return builder(data) if builder else None

def _build_response(summary: str, fmt: str, tool_name: str | None, data) -> dict:
    response: dict = {"answer": summary, "data": data}
    if fmt == "table":
        table = _build_table(tool_name, data)
        if table:
            response["format"] = "table"
            response["table"] = table
            return response
    if fmt == "stats":
        stats = _build_stats(tool_name, data)
        if stats:
            response["format"] = "stats"
            response["stats"] = stats
            return response
    response["format"] = "text"
    return response

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def _parse_final_answer(content: str | None) -> tuple[str, str]:
    if not content:
        return "I couldn't find an answer to that.", "text"
    cleaned = _JSON_FENCE_RE.sub("", content.strip())
    try:
        parsed = json.loads(cleaned)
        summary = parsed.get("summary") or content
        fmt = parsed.get("format") if parsed.get("format") in ("table", "stats", "text") else "text"
        return summary, fmt
    except (json.JSONDecodeError, AttributeError):
        return content, "text"

def _run_with_llm(message: str, history: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-MAX_HISTORY_TURNS:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    providers = llm.get_providers()
    if not providers:
        return _deterministic_ask(message)

    active_idx = 0
    last_data = None
    last_tool_name = None

    for _ in range(MAX_TOOL_TURNS):
        turn = None
        while active_idx < len(providers):
            provider = providers[active_idx]
            try:
                turn = provider.generate_with_tools(messages, TOOLS, max_tokens=2048)
            except QuotaExceededError:
                logger.warning("%s quota exceeded -- failing over", provider.provider_name)
                active_idx += 1
                continue
            if turn is None:
                logger.warning("%s returned no result -- failing over", provider.provider_name)
                active_idx += 1
                continue
            break

        if turn is None:
            return _deterministic_ask(message)

        if not turn["tool_calls"]:
            summary, fmt = _parse_final_answer(turn["content"])
            return _build_response(summary, fmt, last_tool_name, last_data)

        messages.append({"role": "assistant", "content": turn["content"], "tool_calls": turn["tool_calls"]})
        for tc in turn["tool_calls"]:
            result = _dispatch(tc["name"], tc["arguments"])
            last_data = result
            last_tool_name = tc["name"]
            payload = _truncate_for_llm(tc["name"], result)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": json.dumps(payload, default=str)[:10000]})

    return _build_response("I'm having trouble narrowing that down -- try a more specific question.", "text", last_tool_name, last_data)

def ask(message: str, history: list[dict] | None = None) -> dict:
    if not llm.get_providers():
        return _deterministic_ask(message)
    return _run_with_llm(message, history or [])

HELP_TEXT = (
    "I can help with: who's available for a skillset, which pipeline deal needs staffing, "
    "how much of the pipeline is covered, which projects are at risk, current allocation/"
    "utilization, a new-project staffing what-if, or the 6-month outlook. "
    "Try: \"who can cover Python and ETL pipelines starting 2026-07-01?\""
)

def _extract_skillset(message: str) -> str:
    match = re.search(r"(?:for|covering|cover|with skills?|skillset)\s+(.+?)(?:\s+starting|\?|$)", message, re.IGNORECASE)
    return match.group(1).strip() if match else message

def _extract_date(message: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", message)
    return match.group(0) if match else "2026-07-01"

def _deterministic_ask(message: str) -> dict:
    text = message.lower()

    if any(k in text for k in ["coverage", "how much of the pipeline", "pipeline covered"]):
        summary = get_coverage_summary()
        return _build_response(
            f"{summary['redeploy_ready_count']} of {summary['total_demand_rows']} pipeline role-requests are ready to "
            f"redeploy internally; {summary['hire_signal_count']} ({summary['hire_signal_pct']}%) read as a hire signal.",
            "stats", "get_recommendations_coverage_summary", summary,
        )

    if any(k in text for k in ["who can", "who's available", "who is available", "recommend", "staff this", "cover"]):
        skillset = _extract_skillset(message)
        result = get_recommendations(skillset_text=skillset, likely_start_date=_extract_date(message))
        top = result["candidates"][:5]
        if not top:
            return _build_response(f"No one currently has capacity for \"{skillset}\". This is a hire signal.", "text", None, result)
        names = ", ".join(f"{c['employee_id']} ({c['bucket']})" for c in top)
        flag = " This skews toward a hire-vs-redeploy gap." if result["hire_vs_redeploy_flag"] else ""
        return _build_response(f"Top matches for \"{skillset}\": {names}.{flag}", "table", "get_recommendations", result)

    if any(k in text for k in ["risk", "at risk", "health", "overrun", "in trouble"]):
        report = get_health_report()
        high = [r for r in report if r["risk_band"] == "high"]
        if not high:
            return _build_response("No projects are currently flagged high-risk.", "text", None, {"projects": report[:10]})
        names = ", ".join(f"{p['project_code']} ({', '.join(p['root_causes'])})" for p in high[:5])
        return _build_response(f"{len(high)} project(s) are high-risk right now: {names}.", "table", "get_health_report", high)

    if any(k in text for k in ["allocat", "utiliz", "free", "available capacity", "over-allocated", "overallocated"]):
        rows = get_allocation_report()
        over = [r for r in rows if r["utilization_band"] == "over_allocated"]
        ending = [r for r in rows if r["ending_soon"]]
        return _build_response(
            f"{len(over)} allocation rows are over-allocated; {len(ending)} are ending within 30 days and will free up capacity.",
            "stats", "get_allocation_report", rows,
        )

    if any(k in text for k in ["new project", "what if", "what-if", "take this on", "can we staff"]):
        forecast = get_new_project_forecast([{"category": "Build Phase - Tactical Build", "count": 1}])
        return _build_response(
            "Run the New Project Forecast page for a specific scenario (project category, count) -- "
            f"as a quick reference, one generic Tactical Build project right now would have a shortfall of "
            f"{forecast['total_shortfall_headcount']} heads.",
            "stats", "get_new_project_forecast", forecast,
        )

    if any(k in text for k in ["6-month", "six month", "outlook", "pipeline forecast", "next few months"]):
        outlook = get_six_month_outlook()
        warnings = [m for m in outlook["months"] if m["early_warning"]]
        return _build_response(
            f"{len(warnings)} of the next 6 months show a projected shortfall against confirmed demand."
            if warnings else "No early-warning shortfalls in the next 6 months against confirmed demand.",
            "stats", "get_six_month_outlook", outlook,
        )

    if "role mix" in text or "role-mix" in text:
        result = get_role_mix("Client Project", None)
        return _build_response("Here's the org-wide Client Project role-mix template.", "text", None, result)

    return _build_response(HELP_TEXT, "text", None, None)
