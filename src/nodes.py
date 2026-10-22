import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.state import ResearchState, Evidence, PlanResearch, ReviewResult
from src.tools.search import SearchTool

import httpx
from bs4 import BeautifulSoup

import re




load_dotenv()


model = init_chat_model(
    model=os.environ["LLM_MODEL"],
    temperature=0,
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    timeout=120,
    max_retries=0,
)


search_tool = SearchTool(
    api_key=os.environ["TAVILY_API_KEY"]
)

def formate_evidence(state: ResearchState):
    evidence = state["evidence"]
    results = []
    for e in evidence[:8]:
        result =f"""
[S{e.the_id}]
标题：{e.source_title}
来源：{e.source_url}
证据：{e.excerpt[:500]}
"""
        results.append(result)
    return "\n".join(results)


def search_node(state: ResearchState) -> dict:
    if not state["search_query"]:
        return {
            "search_results": [],
            "errors": state.get("errors",[])
            + ["search skipped: search_query is empty"]
        }

    search_query = state["search_query"][-1].strip()
    if not search_query:
        return {
            "search_results": [],
            "errors": state.get("errors",[])
            + ["search skipped: search_query is empty"]
        }

    try:
        search_results = search_tool.search(query=search_query)
        return {
            "search_results": search_results,
            "errors": [],
        }
    except Exception as exc:
        return {
            "search_results": [],
            "errors":[f"Search failed: {exc}"],
        }

def query_generation_node(state: ResearchState) -> dict:
    prompt = f"""
    你是一个搜索查询优化助手。

    请把下面的研究问题改写成适合网页搜索的简洁查询词。
    只返回一条纯文本搜索词，不要返回 JSON，不要解释。

    研究问题：
    {state['question']}
    当前已经查过的关键词(如果是空列表就代表没有):
    {state['search_query']}
    目前已经参考的文章证据，尽可能让关键词不要重复：
    {formate_evidence(state)}
    需要研究的子问题列表，请自己分析根据当前文章证据和关键词，选择重合度不高的方向：
    {state['search_plan']}
    后面审查模型的反馈：
    {state['review_feedback']}
    审查模型指出的缺失话题：
    {state['missing_topics']}
    """

    try:
        print("[make_query]开始调用", flush=True)
        result = model.invoke(prompt)
        print("[make_query]调用结束", flush=True)
        return {
            "search_query": [result.content],
            "errors": []
                }
    except Exception as exc:
        return {
            "search_query": [],
            "errors": [f"Query_generation failed: {exc}"],
            "query_generation_failed": True
        }

def normalize_results_node(state: ResearchState) -> dict:
    raw_results = state["search_results"]
    cleaned_results = []
    seen_urls = set()
    for raw in raw_results:
        url = raw.url.strip()
        content = raw.content.strip()
        if not url:
            continue
        if not content:
            continue
        if url not in seen_urls:
            seen_urls.add(url)
        else:
            continue
        cleaned_results.append(raw.model_copy(update={"url": url, "content": content}))
    return {
        "search_results": cleaned_results,
        "errors": state.get("errors", [])
            }

def fetch_content_node(state: ResearchState) -> dict:
    raw_results = state["search_results"]
    top_5_raw = raw_results[:5]
    new_results = []
    errors = list(state.get("errors",[]))
    for raw in top_5_raw:
        try:
            response = httpx.get(raw.url, timeout=10, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            content = soup.get_text(" ", strip=True)
            new_results.append(raw.model_copy(update={"content": content}))
        except Exception as exc:
            errors.append(f"fetch failed for {raw.url}: {exc}")
            new_results.append(raw)
    return {
        "search_results": new_results,
        "errors": errors
    }

def build_evidence_node(state: ResearchState) -> dict:
    results = state["search_results"]
    evidence_list = []
    existing_evidence = state["evidence"]

    seen_urls = [evid.source_url for evid in existing_evidence]
    the_id = max((e.the_id for e in existing_evidence), default=0)

    for result in results:
        source_url = result.url
        source_title = result.title
        excerpt = result.content[:200]

        if source_url in seen_urls:
            continue

        the_id = the_id + 1
        seen_urls.append(source_url)

        evidence_list.append(
            Evidence(
                source_title=source_title,
                source_url=source_url,
                excerpt=excerpt,
                the_id=the_id,
            )
        )
    return {
        "evidence": evidence_list,
        "turn": state["turn"]-1,
    }

def plan_search_node(state: ResearchState) -> dict:
    question = state["question"]
    prompt = f"""
    你是一个研究规划助手，负责把主问题拆解出2-4个子问题，
    以便后续搜索模型能够获得更全面的信息,
    主问题：{question},
    请严格按照以下 JSON 格式输出，不要添加其他文字：
    {{
    "sub_questions": [
        "子问题1",
        "子问题2",
        "子问题3"
    ]
    }}
    """
    structured_model = model.with_structured_output(
        PlanResearch,
        method="json_mode",
                                                    )
    result = structured_model.invoke(prompt)
    research_plan = result.sub_questions
    return {
        "search_plan": research_plan
    }

def report_generation_node(state: ResearchState) -> dict:
    question = state["question"]
    search_plan = state["search_plan"]
    evidence = formate_evidence(state)
    prompt = f"""
请研究问题：{question}，整理成研究报告的形式。
研究规划：{search_plan}。
当前已有的证据：{evidence}。
请遵循以下要求：
- 根据研究问题组织报告结构。
- 重要事实和结论使用 [S1]、[S2] 等引用 ID。
- 只能引用实际提供的证据 ID，不得编造来源。
- 证据不足时明确说明，不要自行补造事实。
- 结合证据进行综合分析，而不是逐条复述搜索结果。
"""
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Evidence count: {len(state['evidence'])}")
    report = model.invoke(prompt)
    return {
        "report": report.content
    }

def validate_citations_node(state: ResearchState) -> dict:
    report = state["report"]
    citation_ids = {int(x) for x in re.findall(r"\[S(\d+)\]", report)}
    evidence_ids = {e.the_id for e in state["evidence"]}

    invalid_ids = sorted(citation_ids - evidence_ids)
    return {
        "citation_errors": [
            f"引用非法id: {invalid_ids}"
        ]
    }

def review_node(state: ResearchState) -> dict:
    prompt = f"""
    你是一名严谨的研究报告审查员，负责评估研究报告的质量，并为后续研究提供改进建议。

    请根据以下信息进行审查。

    研究问题：
    {state["question"]}

    研究规划：
    {state["search_plan"]}

    当前已有的证据：
    {formate_evidence(state)}

    研究报告：
    {state["report"]}

    请重点评估：
    1. 报告是否充分回答了研究问题，并覆盖研究规划中的主要子问题。
    2. 重要事实和结论是否有现有证据支持，引用是否准确。
    3. 是否存在关键数据缺失、分析不充分或逻辑问题。
    4. 如果报告不合格，具体还需要补充哪些研究方向。

    输出要求：
    - approved：布尔值，报告达到要求时为 true，否则为 false。
    - feedback：字符串，说明审查结论及具体理由。
    - missing_topics：字符串列表，列出需要补充的具体研究方向。
    - 如果报告已经达到要求，missing_topics 返回空列表。
    - 严格返回合法 JSON，不要添加 Markdown 代码块或其他解释文字。"""
    
    structured_model = model.with_structured_output(ReviewResult, method="json_mode")
    result = structured_model.invoke(prompt)

    next_review_round = state["review_round"] + 1
    if (not result.approved) and next_review_round < state["max_review_round"]:
        return {
                "review_approved": result.approved,
                "review_feedback": result.feedback,
                "missing_topics": result.missing_topics,
                "review_round": next_review_round,
                "turn": 2,
        }
    else:
        return {
                "review_approved": result.approved,
                "review_feedback": result.feedback,
                "missing_topics": result.missing_topics,
                "review_round": next_review_round,
        }
