"""Specialized stateful nodes with bounded plan repair."""

import json
from itertools import pairwise
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch, QualityReport, TransformPlan
from agenticdatapipe.operations import execute_plan, profile_records
from agenticdatapipe.storage import persist_batch


class PipelineState(TypedDict, total=False):
    batch: dict[str, Any]
    profile: dict[str, Any]
    plan: dict[str, Any]
    result: dict[str, Any]
    quality: dict[str, Any]
    attempts: int
    report: dict[str, Any]


def build_graph(settings: Settings, fixture: bool = False, planner_override: Any = None):
    """Compile the graph. Overrides are for deterministic failure-path testing."""

    def scout(state: PipelineState) -> dict[str, Any]:
        Batch.model_validate(state["batch"])
        return {"attempts": 0}

    def profiler(state: PipelineState) -> dict[str, Any]:
        return {
            "profile": profile_records.invoke(
                {"records": state["batch"]["records"], "stale_seconds": settings.stale_seconds}
            )
        }

    def planner(state: PipelineState) -> dict[str, Any]:
        if planner_override:
            plan = planner_override(state)
        elif fixture:
            profile = state["profile"]
            plan = TransformPlan(
                allow_bikes_available_alias=bool(profile["alias_candidates"]),
                freshness_grace_seconds=900 if profile["within_freshness_grace"] else 0,
                rationale="Apply the profiled, allowlisted station policies",
            )
        else:
            llm = ChatOpenAI(model=settings.openai_model, temperature=0, max_retries=2, timeout=60)
            structured = llm.with_structured_output(TransformPlan, method="json_schema")
            plan = structured.invoke(
                [
                    (
                        "system",
                        (
                            "Choose bounded policies for bike station observations. Enable the "
                            "bikes_available alias only when alias_candidates is positive; canonical "
                            "num_bikes_available always takes precedence. Choose 900 seconds of "
                            "freshness grace only when within_freshness_grace is positive; otherwise "
                            "choose 0. Use last_reported as event time. Preserve invalid records "
                            "for quarantine and never invent measurements."
                        ),
                    ),
                    (
                        "human",
                        json.dumps(
                            {
                                "profile": state["profile"],
                                "previous_issues": state.get("quality", {}).get("issues", []),
                            }
                        ),
                    ),
                ]
            )
        return {
            "plan": TransformPlan.model_validate(plan).model_dump(),
            "attempts": state["attempts"] + 1,
        }

    def executor(state: PipelineState) -> dict[str, Any]:
        return {
            "result": execute_plan.invoke(
                {
                    "records": state["batch"]["records"],
                    "plan": state["plan"],
                    "stale_seconds": settings.stale_seconds,
                    "profile": state["profile"],
                }
            )
        }

    def validator(state: PipelineState) -> dict[str, Any]:
        result = state["result"]
        quality = QualityReport(
            input_rows=len(state["batch"]["records"]),
            accepted_rows=len(result["accepted"]),
            quarantined_rows=len(result["quarantine"]),
            duplicate_rows=result["duplicates"],
            alias_recovered_rows=result["alias_recovered"],
            grace_accepted_rows=result["grace_accepted"],
            issues=result["issues"],
            repairable=bool(result["issues"]),
        )
        if not quality.accepted_rows:
            quality.warnings.append("No usable station observations in this batch")
        if quality.grace_accepted_rows:
            quality.warnings.append("Some observations were accepted under freshness grace")
        return {"quality": quality.model_dump()}

    def route(state: PipelineState) -> str:
        if state["quality"]["repairable"]:
            return "planner" if state["attempts"] <= settings.repair_limit else "failed"
        return "persist"

    def failed(state: PipelineState) -> dict[str, Any]:
        raise RuntimeError(f"Repair budget exhausted: {state['quality']['issues']}")

    def persist(state: PipelineState) -> dict[str, Any]:
        result = {
            **state["result"],
            "quality": state["quality"],
            "profile": state["profile"],
            "plan": state["plan"],
            "attempts": state["attempts"],
            "mode": "fixture" if fixture else "openai",
        }
        return {
            "report": persist_batch(settings, Batch.model_validate(state["batch"]), result)
        }

    graph = StateGraph(PipelineState)
    for name, function in (
        ("scout", scout),
        ("profiler", profiler),
        ("planner", planner),
        ("executor", executor),
        ("validator", validator),
        ("persist", persist),
        ("failed", failed),
    ):
        graph.add_node(name, function)
    chain = [START, "scout", "profiler", "planner", "executor", "validator"]
    for left, right in pairwise(chain):
        graph.add_edge(left, right)
    graph.add_conditional_edges(
        "validator", route, {n: n for n in ("planner", "persist", "failed")}
    )
    graph.add_edge("persist", END)
    graph.add_edge("failed", END)
    return graph.compile()
