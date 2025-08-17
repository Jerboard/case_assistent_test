from aiogram.types import Message
from aiogram.filters.command import CommandStart
from aiogram.enums.content_type import ContentType

import logging

from init import dp, query_engine


logger = logging.getLogger(__name__)


@dp.message(CommandStart())
async def com_start(msg: Message):
    text = (
        '👋 Привет! Я EORA Assistant — тестовый ИИ-бот компании EORA.\n'
        'Моя задача — помогать вам знакомиться с кейсами и проектами компании.\n\n'
        'Вы можете задать мне вопрос, и я постараюсь ответить, опираясь только на материалы с сайта EORA.'
    )

    await msg.answer(text)



@dp.message()
async def chat(msg: Message):
    if msg.content_type != ContentType.TEXT.value:
        await msg.answer('📝 Не могу прочитать сообщение - отправьте текст')
        return

    try:
        sent = await msg.answer('⏳ Думаю')

        response = query_engine.query(msg.text)

        await sent.edit_text(str(response))

    except Exception as e:
        logger.exception(e, exc_info=True)
        await msg.answer("⚠️ Произошла ошибка. Попробуйте ещё раз.")
