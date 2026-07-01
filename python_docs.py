import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def search_python_docs(query: str, max_results: int = 1):

    with DDGS() as ddgs:

        results = list(
            ddgs.text(
                f"site:docs.python.org {query}",
                max_results=max_results
            )
        )

    if not results:
        return "No official Python documentation found."

    url = results[0]["href"]

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = soup.find_all("p")

    text = "\n".join(
        p.get_text(" ", strip=True)
        for p in paragraphs[:12]
    )

    return f"Official Documentation\nURL: {url}\n\n{text}"