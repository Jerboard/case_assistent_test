import logging

from pathlib import Path
from dataclasses import dataclass
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


logger = logging.getLogger(__name__)


@dataclass
class Assistant:
    system_prompt: str
    vector_data_dir: Path
    model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    similarity_top_k: int = 4
    response_mode: str = "compact"


    def __post_init__(self) -> None:
        """Создаём движок сразу при инициализации."""
        token_counter = TokenCountingHandler()
        Settings.callback_manager = CallbackManager([token_counter])

        Settings.llm = OpenAI(model=self.model)
        Settings.embed_model = OpenAIEmbedding(model=self.embed_model)

        storage_ctx = StorageContext.from_defaults(persist_dir=str(self.vector_data_dir))
        index = load_index_from_storage(storage_ctx)

        self.query_engine = index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            response_mode=self.response_mode,
            prompt=self.system_prompt,
        )

    def query(self, question: str) -> str:
        response = self.query_engine.query(question)
        text = response.response

        if not text:
            return '🤷 Не смог ответить на вопрос или что-то сломалось. Попробуй спросить ещё раз'

        sources = []
        i = 1
        for node in response.source_nodes:
            url = node.metadata.get("url")
            if url:
                sources.append(f'[<a href="{url}">{i}</a>]')
                i += 1

        if sources:
            text += "\n\nИсточники:\n" + " ".join(sources)

        return text



