from aiogram import Dispatcher
from aiogram.types.bot_command import BotCommand
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.base.base_query_engine import BaseQueryEngine

import asyncio
import logging
import uvloop

from config import conf


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)



asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
loop = asyncio.get_event_loop()
bot = Bot(
    token=conf.token,
    loop=loop,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()


def create_query_engine() -> BaseQueryEngine:
    token_counter = TokenCountingHandler()
    Settings.callback_manager = CallbackManager([token_counter])

    Settings.llm = OpenAI(model="gpt-4o-mini", system_prompt=conf.system_prompt_medium)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    storage_ctx = StorageContext.from_defaults(persist_dir=str(conf.vector_data_dir))
    index = load_index_from_storage(storage_ctx)
    return index.as_query_engine(similarity_top_k=4, response_mode="compact")


query_engine: BaseQueryEngine = create_query_engine()


async def set_main_menu():
    await bot.set_my_commands([
        BotCommand(command='start', description='🚀 Старт')

    ])
