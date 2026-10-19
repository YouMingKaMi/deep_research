from tavily import TavilyClient

from src.state import SearchResult

class SearchTool:
    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int=5, ) -> list[SearchResult]:
        res = self.client.search(query=query, max_results=max_results)
        return [
            SearchResult(
                title=result.get("title",""),
                url=result.get("url",""),
                content=result.get("content",""),
                score=result.get("score")
                )
                for result in res.get("results",[])
            ]
