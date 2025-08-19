import logging
import typing as t

from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.program.openai import OpenAIPydanticProgram



logger = logging.getLogger(__name__)


class Source(BaseModel):
    """Единичный источник, используемый в ответе (привязывается по ID к предложениям)."""
    id: int = Field(..., description="ID источника (нумерация с 1)")
    url: str = Field(..., description="URL источника")


class SentenceCite(BaseModel):
    """Одно предложение ответа и список ID источников, релевантных этому предложению."""
    sentence: str = Field(..., description="Полный текст предложения ответа")
    refs: list[int] = Field(default_factory=list, description="ID источников для этого предложения (нумерация с 1)")


class AnswerPayload(BaseModel):
    """Структурированный ответ ассистента: предложения + реестр источников (ID -> URL)."""
    sentences: list[SentenceCite]
    sources: list[Source]


@dataclass
class Assistant:
    persist_dir: Path
    system_prompt: t.Optional[str]

    model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    similarity_top_k: int = 4
    temperature: float = 0.7

    def __post_init__(self) -> None:
        # создаём настройки модели
        llm_kwargs: dict[str, t.Any] = {"model": self.model, "temperature": self.temperature}
        Settings.embed_model = OpenAIEmbedding(model=self.embed_model)

        # загрузка индекса и подготовка ретривера
        storage_ctx = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
        self._index = load_index_from_storage(storage_ctx)
        self._retriever = self._index.as_retriever(similarity_top_k=self.similarity_top_k)

        #  Запрашиваем вернуть строго JSON по нашей схеме
        self._program = OpenAIPydanticProgram(
            tool_choice='required',
            output_cls=AnswerPayload,
            prompt=PromptTemplate(self.system_prompt),
            llm=OpenAI(**llm_kwargs),
        )

    # собираем контекст в один текст
    def _build_context(self, question: str) -> str:
        nodes = self._retriever.retrieve(question)
        chunks: list[str] = []
        for n in nodes:
            # текст чанка
            chunks.append(n.get_content())
            url = n.metadata.get("url")
            if url:
                chunks.append(f"[URL] {url}")
        return "\n\n---\n\n".join(chunks)

    def answer_json(self, question: str) -> AnswerPayload:
        context = self._build_context(question)
        payload: AnswerPayload = self._program(
            question=question,
            context=context,
        )
        return payload

    def answer_text(self, question: str) -> str:
        payload = self.answer_json(question)

        # собираем основной текст. предложение + ссылки
        parts: list[str] = []
        source_dict = {source.id: source.url for source in payload.sources}

        for s in payload.sentences:
            suffix = (" " + "".join(f'[<a href="{source_dict.get(i)}">{i}</a>]' for i in s.refs)) if s.refs else ""
            parts.append(s.sentence.rstrip() + suffix)

        text = " ".join(parts).strip()

        if not text:
            return '🤷 Не смог ответить на вопрос или что-то сломалось. Попробуй спросить ещё раз'

        return text





