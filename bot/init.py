from aiogram import Dispatcher
from aiogram.types.bot_command import BotCommand
from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

import asyncio
import logging
import uvloop

from config import conf, prompts
from objects import Assistant


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
    persist_dir=conf.persist_dir,
    system_prompt=prompts.system_prompt_hard,
)

async def set_main_menu():
    await bot.set_my_commands([
        BotCommand(command='start', description='🚀 Старт')

    ])
