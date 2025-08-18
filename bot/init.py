from aiogram import Dispatcher
from aiogram.types.bot_command import BotCommand
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

import asyncio
import logging
import uvloop

from config import conf
from assistant import Assistant


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)



asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
loop = asyncio.get_event_loop()
bot = Bot(
    token=conf.token,
    loop=loop,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)

dp = Dispatcher()

eora_assist = Assistant(
    system_prompt=conf.system_prompt_hard,
    vector_data_dir=conf.vector_data_dir
)

async def set_main_menu():
    await bot.set_my_commands([
        BotCommand(command='start', description='🚀 Старт')

    ])
