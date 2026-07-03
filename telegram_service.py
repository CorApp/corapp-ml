import os
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.channels import InviteToChannelRequest, CreateChannelRequest
from telethon.errors import FloodWaitError

API_ID = int(os.environ.get("TELEGRAM_API_ID"))
API_HASH = os.environ.get("TELEGRAM_API_HASH")
STRING_SESSION = os.environ.get("TELEGRAM_STRING_SESSION")
ADMIN_ID = int(os.environ.get("TELEGRAM_ADMIN_ID"))
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")


def create_tenant_group(business_name: str) -> dict:
    """
    Crea un supergrupo privado de Telegram para un tenant y agrega
    al bot de notificaciones (@vendelo_notif_bot).

    Puede lanzar FloodWaitError si la cuenta hizo demasiadas
    acciones seguidas — el llamador debe manejarlo.
    """
    with TelegramClient(StringSession(STRING_SESSION), API_ID, API_HASH) as client:
        group_name = f"{business_name}-Vendelo"

        result = client(CreateChannelRequest(
            title=group_name,
            about=f"Grupo de notificaciones Vendelo para {business_name}",
            megagroup=True
        ))
        chat = result.chats[0]
        chat_id = chat.id

        bot_entity = client.get_entity("@vendelo_notif_bot")
        client(InviteToChannelRequest(chat, [bot_entity]))

        return {
            "chat_id": int(f"-100{chat_id}"),
            "group_name": group_name
        }
