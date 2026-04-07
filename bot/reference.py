import os
from aiogram import Bot, Dispatcher, dispatcher
from google import genai
import sys
import logging
import asyncio
from aiogram.filters import CommandStart, Command
from aiogram.types import Message



gemini_key= ""
telegram_key= ""
model = "gemini-3-flash-preview"

class Reference:

    def __init__(self) -> None:
        self.response = ""

reference = Reference()

def clear_past():
    reference.response = ""

dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello!")



@dp.message()
async def gem(message: Message):

    client = genai.Client(api_key="")

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=message.text,
    )
    await message.answer(response.text)


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=telegram_key)

    # And the run events dispatching
    await dp.start_polling(bot)






logging.basicConfig(level=logging.INFO, stream=sys.stdout)
asyncio.run(main())