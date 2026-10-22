from src.graph import graph

initial_state = {
    "question": "AI时代下，各个行业都受到了多大程度的冲击" ,
    "search_query": [],
    "search_results":[],
    "errors":[],
    "evidence":[],
    "turn": 2,
    "search_plan":[],
    "query_generation_failed": False,
    "report": "",
    "review_approved": False,
    "review_feedback": "",
    "missing_topics": [],
    "review_round": 0,
    "max_review_round": 2,
}

# result = graph.invoke(initial_state)

# print("\n搜索规划：", result["search_plan"])
# print("query_generation_fail: ", result["query_generation_failed"])
# print("研究问题：", result["question"])
# print("搜索词：", result["search_query"])
# print("错误信息：", result["errors"])

# print("\n搜索结果：")
# print("报告如下:\n"+ result["report"])
result = graph.invoke(initial_state)

print("\n" + "=" * 60)
print("              Deep Research Agent 运行结果")
print("=" * 60)

# 1. 研究规划
print("\n【1. 研究规划】")
for i, question in enumerate(result.get("search_plan", []), 1):
    print(f"{i}. {question}")

# 2. 搜索过程
print("\n【2. 搜索过程】")
queries = result.get("search_query", [])
print(f"搜索词总数：{len(queries)}")
for i, query in enumerate(queries, 1):
    print(f"  {i}. {query}")

# 3. 搜索结果
search_results = result.get("search_results", [])
print(f"\n【3. 最后一轮搜索结果】共 {len(search_results)} 条")
for i, item in enumerate(search_results, 1):
    print(f"\n  [{i}] {item.title}")
    print(f"  URL: {item.url}")
    print(f"  内容预览: {item.content[:150]}")

# 4. Evidence
evidence = result.get("evidence", [])
print(f"\n【4. 累计证据】共 {len(evidence)} 条")
for item in evidence:
    print(f"\n  [S{item.the_id}] {item.source_title}")
    print(f"  URL: {item.source_url}")
    print(f"  摘要: {item.excerpt[:120]}")

# 5. 引用校验
citation_errors = result.get("citation_errors", [])
print("\n【5. 引用校验】")
if citation_errors:
    for error in citation_errors:
        print(f"  - {error}")
else:
    print("  未记录引用错误（需结合校验节点逻辑确认是否真正通过）")

# 6. Reviewer 审查结果
print("\n【6. Reviewer 审查】")
print(f"是否通过：{result.get('review_approved', False)}")
print(f"审查轮数：{result.get('review_round', 0)}")
print(f"最大审查轮数：{result.get('max_review_round', 2)}")
print(f"审查反馈：{result.get('review_feedback', '')}")

print("需要补充的话题：")
for topic in result.get("missing_topics", []):
    print(f"  - {topic}")

# 7. 错误信息
errors = result.get("errors", [])
print(f"\n【7. 运行错误】共 {len(errors)} 条")
for error in errors:
    print(f"  - {error}")

# 8. 最终报告
print("\n" + "=" * 60)
print("                    最终研究报告")
print("=" * 60)
print(result.get("report", "未生成报告"))

print("\n" + "=" * 60)
print("                    运行结束")
print("=" * 60)
