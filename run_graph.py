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
    "report": ""
}

result = graph.invoke(initial_state)

print("\n搜索规划：", result["search_plan"])
print("query_generation_fail: ", result["query_generation_failed"])
print("研究问题：", result["question"])
print("搜索词：", result["search_query"])
print("错误信息：", result["errors"])

print("\n搜索结果：")
print("报告如下:\n"+ result["report"])
