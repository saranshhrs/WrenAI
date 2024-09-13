import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
from hamilton import base
from hamilton.experimental.h_async import AsyncDriver
from haystack import component
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

from src.core.pipeline import BasicPipeline
from src.core.provider import LLMProvider
from src.pipelines.sql_explanation.components.prompts import (
    sql_explanation_system_prompt,
)
from src.utils import async_timer, timer
from src.web.v1.services.sql_explanation import StepWithAnalysisResults

logger = logging.getLogger("wren-ai-service")


sql_explanation_user_prompt_template = """
### TASK ###

Given the SQL query analysis and other related information, {{ explanation_hint }}

### INPUT ###

Question: {{ question }}
SQL query: {{ sql }}
SQL query summary: {{ sql_summary }}
SQL query analysis: {{ sql_analysis_result }}

Let's think step by step.
"""


def _compose_sql_expression_of_filter_type(
    filter_analysis: Dict, top: bool = True
) -> Dict:
    if filter_analysis["type"] == "EXPR":
        if top:
            return {
                "values": filter_analysis["node"],
                "id": filter_analysis.get("id", ""),
            }
        else:
            return filter_analysis["node"]

    elif filter_analysis["type"] in ("AND", "OR"):
        left_expr = _compose_sql_expression_of_filter_type(
            filter_analysis["left"], top=False
        )
        right_expr = _compose_sql_expression_of_filter_type(
            filter_analysis["right"], top=False
        )
        return {
            "values": f"{left_expr} {filter_analysis['type']} {right_expr}",
            "id": filter_analysis.get("id", ""),
        }

    return {"values": "", "id": ""}


def _compose_sql_expression_of_groupby_type(
    groupby_keys: List[List[dict]],
) -> List[str]:
    return [
        {
            "values": groupby_key["expression"],
            "id": groupby_key.get("id", ""),
        }
        for groupby_key_list in groupby_keys
        for groupby_key in groupby_key_list
    ]


def _compose_sql_expression_of_relation_type(
    relation: Dict, cte_names: List[str]
) -> List[str]:
    def _is_subquery_or_has_subquery_child(relation):
        if relation["type"] == "SUBQUERY":
            return True
        if relation["type"].endswith("_JOIN"):
            if (
                relation["left"]["type"] == "SUBQUERY"
                or relation["right"]["type"] == "SUBQUERY"
            ):
                return True
        return False

    def _collect_relations(relation, result, cte_names, top_level: bool = True):
        if _is_subquery_or_has_subquery_child(relation):
            return

        if relation["type"] == "TABLE" and top_level:
            if relation["tableName"] in cte_names:
                return

            result.append(
                {
                    "values": {
                        "type": relation["type"],
                        "tableName": relation["tableName"],
                    },
                    "id": relation.get("id", ""),
                }
            )
        elif relation["type"].endswith("_JOIN"):
            result.append(
                {
                    "values": {
                        "type": relation["type"],
                        "criteria": relation["criteria"]["expression"],
                        "exprSources": [
                            {
                                "expression": expr_source["expression"],
                                "sourceDataset": expr_source["sourceDataset"],
                                "sourceColumn": expr_source["sourceColumn"],
                            }
                            for expr_source in relation["exprSources"]
                        ],
                    },
                    "id": relation.get("id", ""),
                }
            )
            _collect_relations(relation["left"], cte_names, result, top_level=False)
            _collect_relations(relation["right"], cte_names, result, top_level=False)

    results = []
    _collect_relations(relation, results, cte_names)
    return results


def _compose_sql_expression_of_select_type(
    select_items: List[Dict], selected_data_sources: List[List[Dict]]
) -> Dict:
    def _is_select_item_existed_in_selected_data_sources(
        select_item, selected_data_sources
    ):
        for selected_data_source in selected_data_sources:
            for data_source in selected_data_source:
                if (
                    "exprSources" in select_item
                    and select_item["exprSources"]
                    and select_item["exprSources"][0]["sourceDataset"]
                    == data_source["sourceDataset"]
                    and select_item["exprSources"][0]["sourceColumn"]
                    == data_source["sourceColumn"]
                ):
                    return True
        return False

    result = {
        "withFunctionCallOrMathematicalOperation": [],
        "withoutFunctionCallOrMathematicalOperation": [],
    }

    for select_item in select_items:
        if not _is_select_item_existed_in_selected_data_sources(
            select_item, selected_data_sources
        ):
            if (
                select_item["properties"]["includeFunctionCall"] == "true"
                or select_item["properties"]["includeMathematicalOperation"] == "true"
            ):
                result["withFunctionCallOrMathematicalOperation"].append(
                    {
                        "values": {
                            "alias": select_item["alias"],
                            "expression": select_item["expression"],
                        },
                        "id": select_item.get("id", ""),
                    }
                )
            else:
                result["withoutFunctionCallOrMathematicalOperation"].append(
                    {
                        "values": {
                            "alias": select_item["alias"],
                            "expression": select_item["expression"],
                        },
                        "id": select_item.get("id", ""),
                    }
                )

    return result


def _compose_sql_expression_of_sortings_type(sortings: List[Dict]) -> List[str]:
    return [
        {
            "values": f'{sorting["expression"]} {sorting["ordering"]}',
            "id": sorting.get("id", ""),
        }
        for sorting in sortings
    ]


def get_sql_explanation_prompt_given_analysis_type(sql_analysis_type: str) -> str:
    if sql_analysis_type == "filter":
        return "Explain the filter condition in the SQL query."
    elif sql_analysis_type == "groupByKeys":
        return "Explain the group by keys in the SQL query."
    elif sql_analysis_type == "relation":
        return "Explain the relation in the SQL query."
    elif sql_analysis_type == "selectItems":
        return "Explain the select items in the SQL query."
    elif sql_analysis_type == "sortings":
        return "Explain the sortings in the SQL query."
    return ""


def _extract_to_str(data):
    if isinstance(data, list) and data:
        return data[0]
    elif isinstance(data, str):
        return data

    return ""


@component
class SQLAnalysisPreprocessor:
    @component.output_types(
        preprocessed_sql_analysis_results=List[Dict],
    )
    def run(
        self,
        cte_names: List[str],
        selected_data_sources: List[List[Dict]],
        sql_analysis_results: List[Dict],
    ) -> Dict[str, List[Dict]]:
        preprocessed_sql_analysis_results = []
        for sql_analysis_result in sql_analysis_results:
            if not sql_analysis_result.get("isSubqueryOrCte", False):
                preprocessed_sql_analysis_result = {}
                if "filter" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "filter"
                    ] = _compose_sql_expression_of_filter_type(
                        sql_analysis_result["filter"]
                    )
                else:
                    preprocessed_sql_analysis_result["filter"] = {}
                if "groupByKeys" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "groupByKeys"
                    ] = _compose_sql_expression_of_groupby_type(
                        sql_analysis_result["groupByKeys"]
                    )
                else:
                    preprocessed_sql_analysis_result["groupByKeys"] = []
                if "relation" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "relation"
                    ] = _compose_sql_expression_of_relation_type(
                        sql_analysis_result["relation"],
                        cte_names,
                    )
                else:
                    preprocessed_sql_analysis_result["relation"] = []
                if "selectItems" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "selectItems"
                    ] = _compose_sql_expression_of_select_type(
                        sql_analysis_result["selectItems"], selected_data_sources
                    )
                else:
                    preprocessed_sql_analysis_result["selectItems"] = {
                        "withFunctionCallOrMathematicalOperation": [],
                        "withoutFunctionCallOrMathematicalOperation": [],
                    }
                if "sortings" in sql_analysis_result:
                    preprocessed_sql_analysis_result[
                        "sortings"
                    ] = _compose_sql_expression_of_sortings_type(
                        sql_analysis_result["sortings"]
                    )
                else:
                    preprocessed_sql_analysis_result["sortings"] = []
                preprocessed_sql_analysis_results.append(
                    preprocessed_sql_analysis_result
                )

        return {"preprocessed_sql_analysis_results": preprocessed_sql_analysis_results}


@component
class GenerationPostProcessor:
    @component.output_types(
        results=Optional[List[Dict[str, Any]]],
    )
    def run(
        self, generates: List[List[str]], preprocessed_sql_analysis_results: List[dict]
    ) -> Dict[str, Any]:
        results = []
        try:
            if preprocessed_sql_analysis_results:
                preprocessed_sql_analysis_results = preprocessed_sql_analysis_results[0]
                for generate in generates:
                    sql_explanation_results = orjson.loads(generate["replies"][0])[
                        "results"
                    ]

                    logger.debug(
                        f"sql_explanation_results: {orjson.dumps(sql_explanation_results, option=orjson.OPT_INDENT_2).decode()}"
                    )

                    if preprocessed_sql_analysis_results.get(
                        "filter", {}
                    ) and sql_explanation_results.get("filter", {}):
                        results.append(
                            {
                                "type": "filter",
                                "payload": {
                                    "id": preprocessed_sql_analysis_results["filter"][
                                        "id"
                                    ],
                                    "expression": preprocessed_sql_analysis_results[
                                        "filter"
                                    ]["values"],
                                    "explanation": _extract_to_str(
                                        sql_explanation_results["filter"]
                                    ),
                                },
                            }
                        )
                    elif preprocessed_sql_analysis_results.get(
                        "groupByKeys", []
                    ) and sql_explanation_results.get("groupByKeys", []):
                        for (
                            groupby_key,
                            sql_explanation,
                        ) in zip(
                            preprocessed_sql_analysis_results["groupByKeys"],
                            sql_explanation_results["groupByKeys"],
                        ):
                            results.append(
                                {
                                    "type": "groupByKeys",
                                    "payload": {
                                        "id": groupby_key["id"],
                                        "expression": groupby_key["values"],
                                        "explanation": _extract_to_str(sql_explanation),
                                    },
                                }
                            )
                    elif preprocessed_sql_analysis_results.get(
                        "relation", []
                    ) and sql_explanation_results.get("relation", []):
                        for (
                            relation,
                            sql_explanation,
                        ) in zip(
                            preprocessed_sql_analysis_results["relation"],
                            sql_explanation_results["relation"],
                        ):
                            results.append(
                                {
                                    "type": "relation",
                                    "payload": {
                                        "id": relation["id"],
                                        **relation["values"],
                                        "explanation": _extract_to_str(sql_explanation),
                                    },
                                }
                            )
                    elif preprocessed_sql_analysis_results.get(
                        "selectItems", {}
                    ) and sql_explanation_results.get("selectItems", {}):
                        sql_analysis_result_for_select_items = [
                            {
                                "type": "selectItems",
                                "payload": {
                                    "id": select_item["id"],
                                    **select_item["values"],
                                    "isFunctionCallOrMathematicalOperation": True,
                                    "explanation": _extract_to_str(sql_explanation),
                                },
                            }
                            for select_item, sql_explanation in zip(
                                preprocessed_sql_analysis_results.get(
                                    "selectItems", {}
                                ).get("withFunctionCallOrMathematicalOperation", []),
                                sql_explanation_results.get("selectItems", {}).get(
                                    "withFunctionCallOrMathematicalOperation", []
                                ),
                            )
                        ] + [
                            {
                                "type": "selectItems",
                                "payload": {
                                    "id": select_item["id"],
                                    **select_item["values"],
                                    "isFunctionCallOrMathematicalOperation": False,
                                    "explanation": _extract_to_str(sql_explanation),
                                },
                            }
                            for select_item, sql_explanation in zip(
                                preprocessed_sql_analysis_results.get(
                                    "selectItems", {}
                                ).get("withoutFunctionCallOrMathematicalOperation", []),
                                sql_explanation_results.get("selectItems", {}).get(
                                    "withoutFunctionCallOrMathematicalOperation", []
                                ),
                            )
                        ]

                        results += sql_analysis_result_for_select_items
                    elif preprocessed_sql_analysis_results.get(
                        "sortings", []
                    ) and sql_explanation_results.get("sortings", []):
                        for (
                            sorting,
                            sql_explanation,
                        ) in zip(
                            preprocessed_sql_analysis_results["sortings"],
                            sql_explanation_results["sortings"],
                        ):
                            results.append(
                                {
                                    "type": "sortings",
                                    "payload": {
                                        "id": sorting["id"],
                                        "expression": sorting["values"],
                                        "explanation": _extract_to_str(sql_explanation),
                                    },
                                }
                            )
        except Exception as e:
            logger.exception(f"Error in GenerationPostProcessor: {e}")

        return {"results": results}


## Start of Pipeline
@timer
@observe(capture_input=False)
def preprocess(
    selected_data_sources: List[List[dict]],
    sql_analysis_results: List[dict],
    cte_names: List[str],
    pre_processor: SQLAnalysisPreprocessor,
) -> dict:
    logger.debug(
        f"sql_analysis_results: {orjson.dumps(sql_analysis_results, option=orjson.OPT_INDENT_2).decode()}"
    )
    return pre_processor.run(cte_names, selected_data_sources, sql_analysis_results)


@timer
@observe(capture_input=False)
def prompts(
    question: str,
    sql: str,
    preprocess: dict,
    sql_summary: str,
    prompt_builder: PromptBuilder,
) -> List[dict]:
    logger.debug(f"question: {question}")
    logger.debug(f"sql: {sql}")
    logger.debug(
        f"preprocess: {orjson.dumps(preprocess, option=orjson.OPT_INDENT_2).decode()}"
    )
    logger.debug(f"sql_summary: {sql_summary}")

    sql_analysis_types = []
    preprocessed_sql_analysis_results_with_values = []
    for preprocessed_sql_analysis_result in preprocess[
        "preprocessed_sql_analysis_results"
    ]:
        for key, value in preprocessed_sql_analysis_result.items():
            if value:
                sql_analysis_types.append(key)
                if key != "selectItems":
                    if isinstance(value, list):
                        preprocessed_sql_analysis_results_with_values.append(
                            {key: [v["values"] for v in value]}
                        )
                    else:
                        preprocessed_sql_analysis_results_with_values.append(
                            {key: value["values"]}
                        )
                else:
                    preprocessed_sql_analysis_results_with_values.append(
                        {
                            key: {
                                "withFunctionCallOrMathematicalOperation": [
                                    v["values"]
                                    for v in value[
                                        "withFunctionCallOrMathematicalOperation"
                                    ]
                                ],
                                "withoutFunctionCallOrMathematicalOperation": [
                                    v["values"]
                                    for v in value[
                                        "withoutFunctionCallOrMathematicalOperation"
                                    ]
                                ],
                            }
                        }
                    )

    logger.debug(
        f"preprocessed_sql_analysis_results_with_values: {orjson.dumps(preprocessed_sql_analysis_results_with_values, option=orjson.OPT_INDENT_2).decode()}"
    )

    return [
        prompt_builder.run(
            question=question,
            sql=sql,
            sql_analysis_result=sql_analysis_result,
            explanation_hint=get_sql_explanation_prompt_given_analysis_type(
                sql_analysis_type
            ),
            sql_summary=sql_summary,
        )
        for sql_analysis_type, sql_analysis_result in zip(
            sql_analysis_types, preprocessed_sql_analysis_results_with_values
        )
    ]


@async_timer
@observe(as_type="generation", capture_input=False)
async def generate_sql_explanation(prompts: List[dict], generator: Any) -> List[dict]:
    logger.debug(
        f"prompts: {orjson.dumps(prompts, option=orjson.OPT_INDENT_2).decode()}"
    )

    async def _task(prompt: str, generator: Any):
        return await generator.run(prompt=prompt.get("prompt"))

    tasks = [_task(prompt, generator) for prompt in prompts]
    return await asyncio.gather(*tasks)


@timer
@observe(capture_input=False)
def post_process(
    generate_sql_explanation: List[dict],
    preprocess: dict,
    post_processor: GenerationPostProcessor,
) -> dict:
    logger.debug(
        f"generate_sql_explanation: {orjson.dumps(generate_sql_explanation, option=orjson.OPT_INDENT_2).decode()}"
    )
    logger.debug(
        f"preprocess: {orjson.dumps(preprocess, option=orjson.OPT_INDENT_2).decode()}"
    )

    return post_processor.run(
        generate_sql_explanation,
        preprocess["preprocessed_sql_analysis_results"],
    )


## End of Pipeline


class Generation(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
    ):
        self.pre_processor = SQLAnalysisPreprocessor()
        self.prompt_builder = PromptBuilder(
            template=sql_explanation_user_prompt_template
        )
        self.generator = llm_provider.get_generator(
            system_prompt=sql_explanation_system_prompt
        )
        self.post_processor = GenerationPostProcessor()

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    def visualize(
        self,
        question: str,
        cte_names: List[str],
        selected_data_sources: List[List[dict]],
        step_with_analysis_results: StepWithAnalysisResults,
    ) -> None:
        destination = "outputs/pipelines/sql_explanation"
        if not Path(destination).exists():
            Path(destination).mkdir(parents=True, exist_ok=True)

        self._pipe.visualize_execution(
            ["post_process"],
            output_file_path=f"{destination}/generation.dot",
            inputs={
                "question": question,
                "cte_names": cte_names,
                "selected_data_sources": selected_data_sources,
                "sql": step_with_analysis_results.sql,
                "sql_analysis_results": step_with_analysis_results.sql_analysis_results,
                "sql_summary": step_with_analysis_results.summary,
                "pre_processor": self.pre_processor,
                "prompt_builder": self.prompt_builder,
                "generator": self.generator,
                "post_processor": self.post_processor,
            },
            show_legend=True,
            orient="LR",
        )

    @async_timer
    @observe(name="SQL Explanation Generation")
    async def run(
        self,
        question: str,
        cte_names: List[str],
        selected_data_sources: List[List[dict]],
        step_with_analysis_results: StepWithAnalysisResults,
    ):
        logger.info("SQL Explanation Generation pipeline is running...")

        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "question": question,
                "cte_names": cte_names,
                "selected_data_sources": selected_data_sources,
                "sql": step_with_analysis_results.sql,
                "sql_analysis_results": step_with_analysis_results.sql_analysis_results,
                "sql_summary": step_with_analysis_results.summary,
                "pre_processor": self.pre_processor,
                "prompt_builder": self.prompt_builder,
                "generator": self.generator,
                "post_processor": self.post_processor,
            },
        )


if __name__ == "__main__":
    from langfuse.decorators import langfuse_context

    from src.core.engine import EngineConfig
    from src.core.pipeline import async_validate
    from src.utils import init_langfuse, init_providers, load_env_vars

    load_env_vars()
    init_langfuse()

    llm_provider, _, _, _ = init_providers(EngineConfig())
    pipeline = Generation(
        llm_provider=llm_provider,
    )

    pipeline.visualize(
        "this is a test question",
        ["xxx"],
        StepWithAnalysisResults(
            sql="xxx",
            summary="xxx",
            cte_name="xxx",
            sql_analysis_results=[],
        ),
    )
    async_validate(
        lambda: pipeline.run(
            "this is a test question",
            ["xxx"],
            StepWithAnalysisResults(
                sql="xxx",
                summary="xxx",
                cte_name="xxx",
                sql_analysis_results=[],
            ),
        )
    )

    langfuse_context.flush()
