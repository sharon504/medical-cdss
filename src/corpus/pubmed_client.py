"""Thin client for NCBI E-utilities (esearch + efetch). Public, no API key:
https://www.ncbi.nlm.nih.gov/books/NBK25501/

Respects the 3 req/s rate limit for key-less access with a sleep interval.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class Abstract:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    source: str  # provenance tag, e.g. "pubmed"

    def as_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "journal": self.journal,
            "year": self.year,
            "source": self.source,
        }


class PubmedClient:
    def __init__(self, sleep_seconds: float = 0.4, timeout: int = 30):
        self.sleep = sleep_seconds
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "medical-cdss/0.1 (educational project)"}
        )

    def _get(self, url: str, params: dict) -> str:
        time.sleep(self.sleep)
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def search(self, query: str, limit: int = 50) -> list[str]:
        """Return up to `limit` PMIDs matching a term."""
        xml = self._get(
            f"{EUTILS}/esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json",
                "sort": "relevance",
            },
        )
        data = json_loads(xml)
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_abstracts(self, pmids: list[str]) -> list[Abstract]:
        """Fetch structured abstracts for a batch of PMIDs."""
        if not pmids:
            return []
        xml = self._get(
            f"{EUTILS}/efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
            },
        )
        return parse_pubmed_xml(xml)


def json_loads(text: str) -> dict:
    import json

    return json.loads(text)


def parse_pubmed_xml(xml: str) -> list[Abstract]:
    root = ET.fromstring(xml)
    out: list[Abstract] = []

    for article in root.iter("PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//Abstract/AbstractText")
        journal_el = article.find(".//Journal/Title")
        year_el = article.find(".//JournalIssue/PubDate/Year")

        abstract_text = "\n".join(
            ("".join(part.itertext())).strip() for part in abstract_parts
            if "".join(part.itertext()).strip()
        )

        out.append(
            Abstract(
                pmid=pmid_el.text if pmid_el is not None else "",
                title="".join(title_el.itertext()) if title_el is not None else "",
                abstract=abstract_text,
                journal=journal_el.text if journal_el is not None else "",
                year=year_el.text if year_el is not None else "",
                source="pubmed",
            )
        )
    return out
