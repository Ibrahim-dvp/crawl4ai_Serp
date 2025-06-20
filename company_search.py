import asyncio
import json
import os
from typing import Optional, List

from pydantic import BaseModel
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
    LLMConfig,
)


class CompanyInfo(BaseModel):
    company_name: Optional[str] = None
    legal_name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    registration_number: Optional[str] = None


async def duckduckgo_search(query: str, *, headless: bool = True) -> List[str]:
    browser_config = BrowserConfig(headless=headless, java_script_enabled=True)
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector="div.results",
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        from urllib.parse import quote
        # Use the static HTML version of DuckDuckGo results to avoid dynamic content
        url = f"https://duckduckgo.com/html/?q={quote(query)}&kl=us-en"
        result = await crawler.arun(url, config=crawler_config)
        if not result.success:
            return []
        extraction_strategy = JsonCssExtractionStrategy(
            schema={
                "name": "results",
                "baseSelector": "div.result",
                "fields": [
                    {"name": "title", "selector": "h2 a", "type": "text"},
                    {
                        "name": "link",
                        "selector": "a.result__a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            }
        )
        items = extraction_strategy.run(url="", sections=[result.html])
        links = []
        for item in items:
            link = item.get("link")
            if not link:
                continue
            from urllib.parse import urlparse, parse_qs
            if link.startswith("//"):
                parsed = urlparse("https:" + link)
                actual = parse_qs(parsed.query).get("uddg", [None])[0]
                if actual:
                    link = actual
                else:
                    link = parsed.geturl()
            links.append(link)
        return links


async def extract_legal_info(url: str, *, headless: bool = True) -> dict:
    browser_config = BrowserConfig(headless=headless)
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="openai/gpt-4o",
                api_token=os.getenv("OPENAI_API_KEY", ""),
            ),
            schema=CompanyInfo.model_json_schema(),
            extraction_type="schema",
            instruction=(
                "Extract the company name, legal name, website, address, phone, "
                "email, and registration number from this page if available."
            ),
        ),
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url, config=crawl_config)
        if result.success and result.extracted_content:
            info = json.loads(result.extracted_content)
            if isinstance(info, list) and info:
                return info[0]
            return info
    return {}


async def search_company(name: str, *, headless: bool = True) -> dict:
    links = await duckduckgo_search(name, headless=headless)
    if not links:
        return {}
    return await extract_legal_info(links[0], headless=headless)


async def main(name: str, *, headless: bool = True):
    info = await search_company(name, headless=headless)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search company information")
    parser.add_argument("name", help="Company name")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run browser in headless mode",
    )
    args = parser.parse_args()

    asyncio.run(main(args.name, headless=args.headless))
