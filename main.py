# main.py
from telethon import TelegramClient, events
from handlers import register_handlers

# Configuration
api_id = 6298540
api_hash = '280c5baa78f9185aba17a0f496302690'
bot_token = "7208547475:AAEsEsc6JZnLuxXMFoPpmA9sqBgd0XpyJns"

# Initialize Telegram Client
client = TelegramClient('gemini_bot', api_id, api_hash).start(bot_token=bot_token)

# Register all handlers through the central registration function
register_handlers(client)

if __name__ == '__main__':
    print("✅ ربات در حال اجراست...")
    client.run_until_disconnected()