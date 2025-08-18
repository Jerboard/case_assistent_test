import time
import requests
import logging

from pathlib import Path
from dataclasses import dataclass
from requests import Session

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from llama_index.core import Document, VectorStoreIndex


logger = logging.getLogger(__name__)


@dataclass
class EoraCaseIndexer:
    source_urls_path: Path
    vector_data_dir: Path
    ua: UserAgent = UserAgent()

    # создаём и очищаем суп
    def _create_requests_session(self) -> Session:
        session = Session()
        session.headers.update({
            "User-Agent": self.ua.chrome
        })
        return session

    # создаём и очищаем суп
    def _create_soup(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "lxml")

        # очищаем от лишнего
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        return soup

    # получаем название кейса
    def _extract_title(self, soup: BeautifulSoup) -> str:
        title = soup.find("title")
        return title.get_text(strip=True) if title else None

    # собираем текстовые данные со страницы
    def _extract_text(self, soup: BeautifulSoup) -> str:

        scope = soup.find("main") or soup.body or soup
        parts: list[str] = []
        for el in scope.find_all(["h1", "h2", "h3", "p", "li"]):
            txt = el.get_text(" ", strip=True)
            if txt:
                parts.append(txt)

        text = "\n".join(parts)
        text = "\n".join(line for line in (ln.strip() for ln in text.splitlines()) if line)
        return text

    def _create_vector(self, docs: list[Document]) -> None:
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(str(self.vector_data_dir))

    # подготоавливаем хранилище при старте бота
    def fetch_pages(self) -> None:
        urls: list[str] = [
            line.strip()
            for line in self.source_urls_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        session = self._create_requests_session()
        docs = []
        for url in urls:
            if not url:
                continue
            logger.warning(f"Fetching {url}")
            try:
                response = session.get(url)
                logger.warning(f"Response {response.status_code}")
                response.raise_for_status()

                # имя файла из последнего сегмента пути

                soup = self._create_soup(response.text)
                page_text = self._extract_text(soup)

                if not page_text:
                    continue

                slug = url.split('/')[-1]
                metadata = {
                    'url': url,
                    'slug': slug,
                    'title': self._extract_title(soup),

                }
                logger.warning(metadata)
                docs.append(Document(text=page_text, metadata=metadata, doc_id=slug))
                time.sleep(1)

            except Exception as e:
                logger.warning(e, exc_info=True)
                return

        if docs:
            self._create_vector(docs)



