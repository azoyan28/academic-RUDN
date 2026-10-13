#!/bin/zsh
set -e
telegram_project_dir="${0:A:h}"
cd "$telegram_project_dir"
if [[ -x "$telegram_project_dir/.venv/bin/python" ]]; then
  telegram_python="$telegram_project_dir/.venv/bin/python"
elif [[ -x "$telegram_project_dir/../../work/venv/bin/python" ]]; then
  telegram_python="$telegram_project_dir/../../work/venv/bin/python"
else
  print 'Сначала выполните установку из README.md.'
  read 'telegram_reply?Нажмите Enter для закрытия '
  exit 1
fi
print 'Telegram-бот использует ту же базу, что и веб-приложение.'
print 'При первом запуске вставьте токен BotFather в скрытый ввод.'
"$telegram_python" -m src.telegram_bot --configure-if-needed
