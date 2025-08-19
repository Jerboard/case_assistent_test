import asyncio
import logging
import sys

from init import set_main_menu, bot
from config import conf
from objects import EoraCaseIndexer

from handlers import dp


logger = logging.getLogger(__name__)

async def main() -> None:
    await set_main_menu()
    await dp.start_polling(bot)
    await bot.session.close()


if __name__ == "__main__":
    if conf.debug:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    else:
        logger.info('start bot')
    asyncio.run(main())
