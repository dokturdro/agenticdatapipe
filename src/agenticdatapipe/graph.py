"""Specialized stateful nodes with bounded plan repair."""

import json
from itertools import pairwise
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import REQUIRED_OPERATIONS, Batch, QualityReport, TransformPlan
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
        return {"profile": profile_records.invoke({"records": state["batch"]["records"]})}

    def planner(state: PipelineState) -> dict[str, Any]:
        if planner_override:
            plan = planner_override(state)
        elif fixture:
            plan = TransformPlan(
                operations=REQUIRED_OPERATIONS, rationale="Apply canonical station contract"
            )
        else:
            llm = ChatOpenAI(model=settings.openai_model, temperature=0, max_retries=2, timeout=60)
            structured = llm.with_structured_output(TransformPlan, method="json_schema")
            plan = structured.invoke(
                [
                    (
                        "system",
                        "Plan bike station transformations. Use all required operations in canonical order: "
                        + ", ".join(REQUIRED_OPERATIONS)
                        + ". Use last_reported as event time, preserve invalid records for quarantine, "
                        "and never invent measurements.",
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
            issues=result["issues"],
            repairable=bool(result["issues"]),
        )
        if not quality.accepted_rows:
            quality.warnings.append("No usable station observations in this batch")
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
