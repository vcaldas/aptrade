#!/bin/bash

# Inicia o bot do Telegram em background
python /app/bot_telegram/telegram_server.py &

# Inicia o aplicativo FastAPI
uvicorn fast_api.main:app --host 0.0.0.0 --port 8080 --reload
