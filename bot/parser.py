import time
import requests
import logging

from pathlib import Path
from dataclasses import dataclass

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from llama_index.core import Document, VectorStoreIndex

logger = logging.getLogger(__name__)


ua = UserAgent()


@dataclass
class EoraCaseIndexer:
    source_urls_path: Path
    vector_data_dir: Path

    # собираем данные со страницы
    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

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

        docs = []

        session = requests.Session()
        session.headers.update({
            "User-Agent": ua.chrome
        })

        for url in urls:
            logger.warning(f"Fetching {url}")
            try:
                response = session.get(url)
                logger.warning(f"Response {response.status_code} {response.text}")
                response.raise_for_status()

                # имя файла из последнего сегмента пути
                page_dict = {}
                parsed = urlparse(url)
                page_dict['url'] = url
                name = parsed.path
                page_text = self._extract_text(response.text)

                if not page_text:
                    continue

                metadata = {
                    'url': url
                }
                docs.append(Document(text=page_text, metadata=metadata, doc_id=name))
                time.sleep(1)

            except Exception as e:
                logger.warning(f"[FAIL] {url}: {e}")
                return

        if docs:
            self._create_vector(docs)



