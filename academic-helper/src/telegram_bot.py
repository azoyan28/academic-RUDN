"""Telegram-интерфейс общей SQLite и модели. python -m src.telegram_bot."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from src.agent import Agent, Tools
from src.config import ARTIFACTS, DATA, DB_PATH, COURSES

REPOSITORY = 'https://github.com/azoyan28/academic-RUDN'
CONFIG = DATA / 'telegram_config.json'
STATE = DATA / 'telegram_state.json'
MENU = [['Статистика', 'Прогноз'], ['Группа риска', 'Справка'], ['Качество модели', 'Репозиторий'], ['Режим ответа', 'Отмена']]
KEYBOARD = {'keyboard': MENU, 'resize_keyboard': True}
DISCLOSURE = 'Учебные данные вымышленного российского вуза, оценки 2–5. Вопросы сохраняются в журнале проекта.'
STEPS = [
 ('course', 'Выберите дисциплину.', list(COURSES.values())),
 ('control_1', 'Оценка за контрольную 1? Допустимы 2, 3, 4, 5.', ['2','3','4','5']),
 ('control_2', 'Оценка за контрольную 2? Допустимы 2, 3, 4, 5.', ['2','3','4','5']),
 ('classes_total', 'Сколько занятий проведено к моменту прогноза? Например, 16.', None),
 ('classes_attended', 'Сколько из них посещено?', None),
 ('labs_assigned', 'Сколько лабораторных назначено? Например, 4.', None),
 ('labs_submitted', 'Сколько из них сдано?', None),
 ('late_labs', 'Сколько сданных работ были сданы с опозданием?', None),
]
HELP = ('Я отвечаю по той же базе, что и веб-приложение.\n\n'
        '/stats — статистика; можно добавить ud, bd или pr\n'
        '/predict — пошаговый прогноз\n'
        '/predict ud-0001 — прогноз для записи из базы\n'
        '/risk bd — список записей повышенного риска\n'
        '/quality — результаты тестирования модели\n'
        '/repo — исходный код и последнее изменение GitHub\n'
        '/mode — режим ответа\n/cancel — отменить ввод\n\n'
        'Можно задать обычный вопрос: «Какая шкала оценок используется?»\n\n'+DISCLOSURE)


def private_json(path, value):
    """Файл создаётся с правами 600, запись заменяется атомарно."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False)
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def valid_token(value):
    return bool(re.fullmatch(r'\d{5,}:[A-Za-z0-9_-]{20,}', value or ''))


class TelegramError(RuntimeError):
    def __init__(self, code=0, retry_after=0):
        self.code, self.retry_after = code, retry_after
        super().__init__('Ошибка Telegram API, код '+str(code))


class TelegramAPI:
    def __init__(self, token, opener=urlopen):
        if not valid_token(token): raise ValueError('Некорректный формат токена BotFather')
        self._token, self.opener = token, opener

    def call(self, method, payload=None):
        allowed = {'getMe','getWebhookInfo','getUpdates','sendMessage','setMyCommands'}
        if method not in allowed: raise ValueError('Недопустимый метод Telegram')
        request = Request('https://api.telegram.org/bot'+self._token+'/'+method,
            data=json.dumps(payload or {}, ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json'}, method='POST')
        try:
            with self.opener(request, timeout=40) as response: data=json.load(response)
        except HTTPError as error:
            try: data=json.loads(error.read())
            except Exception: data={}
            raise TelegramError(error.code, data.get('parameters',{}).get('retry_after',0)) from None
        except (URLError, TimeoutError, OSError, ValueError):
            # Нельзя печатать исходное исключение: URL содержит токен.
            raise TelegramError() from None
        if not data.get('ok'):
            raise TelegramError(data.get('error_code',0), data.get('parameters',{}).get('retry_after',0))
        return data['result']

    def send(self, chat_id, text, keyboard=KEYBOARD):
        for part in split_message(text):
            self.call('sendMessage', {'chat_id':chat_id,'text':part,'reply_markup':keyboard,
                                      'link_preview_options':{'is_disabled':True}})


def split_message(text, limit=3500):
    """Не разрывает Unicode-символ и остаётся ниже лимита Telegram."""
    text=re.sub(r'\*\*(.*?)\*\*', r'\1', str(text), flags=re.S)
    chunks=[];current=[];units=0
    for char in text:
        size=len(char.encode('utf-16-le'))//2
        if units+size>limit:
            chunks.append(''.join(current));current=[];units=0
        current.append(char);units+=size
    if current:chunks.append(''.join(current))
    return chunks or ['Нет ответа.']


def repository_status():
    request=Request('https://api.github.com/repos/azoyan28/academic-RUDN/commits/main',
                    headers={'User-Agent':'academic-helper-telegram','Accept':'application/vnd.github+json'})
    try:
        with urlopen(request,timeout=5) as response: info=json.load(response)
        return (REPOSITORY+'\n\nПоследнее изменение кода: '+info['sha'][:7]+
                '\nДата: '+info['commit']['committer']['date']+
                '\n'+info['commit']['message'].splitlines()[0][:150]+
                '\n\nGitHub хранит исходники. Бот работает на запущенном Mac; обновления кода устанавливаются отдельно.')
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError):
        return REPOSITORY+'\n\nИсходники доступны по ссылке. Сейчас не удалось проверить последнее изменение.'


def local_available():
    import importlib.util
    import platform
    from src.local_llm import MODEL_PATH
    return platform.machine()=='arm64' and importlib.util.find_spec('mlx_lm') is not None and (MODEL_PATH/'model.safetensors').exists()


class Controller:
    def __init__(self, tools=None, default_mode='offline', repo_fetch=repository_status):
        if default_mode not in ('offline','local'):raise ValueError('Режим: offline или local')
        self.tools=tools or Tools();self.default_mode=default_mode;self.repo_fetch=repo_fetch
        self.forms={};self.modes={}

    def reply(self, text, keyboard=KEYBOARD): return {'text':text,'keyboard':keyboard}

    def log(self, question, answer, tool, latency=0):
        with sqlite3.connect(self.tools.db_path,timeout=30) as db:
            db.execute('INSERT INTO requests(created_at,mode,tool_names,question,answer,latency_ms,status) VALUES (?,?,?,?,?,?,?)',
                (datetime.now(timezone.utc).isoformat(),'telegram:offline',json.dumps([tool]),question,answer,latency,'ok'))

    def ask(self, chat_id, question):
        result=Agent(mode=self.modes.get(chat_id,self.default_mode),tools=self.tools).ask(question,channel='telegram')
        return self.reply(result['answer'])

    def step_prompt(self, form):
        _,prompt,choices=STEPS[form['step']]
        keyboard={'keyboard':[[c] for c in choices]+[['Отмена']],'resize_keyboard':True} if choices else {'keyboard':[['Отмена']],'resize_keyboard':True}
        return self.reply(prompt,keyboard)

    def step(self, chat_id, text):
        form=self.forms[chat_id]
        if time.monotonic()-form['updated']>1800:
            del self.forms[chat_id];return self.reply('Время ввода истекло. Начните новый прогноз кнопкой «Прогноз».')
        field,_,_=STEPS[form['step']]; values=form['values']
        if field=='course':
            value=next((code for code,name in COURSES.items() if text.casefold() in (code,name.casefold())),None)
            if value is None:return self.reply('Выберите дисциплину кнопкой или введите ud, bd, pr.',self.step_prompt(form)['keyboard'])
        else:
            if not re.fullmatch(r'\d{1,4}',text):return self.reply('Введите целое неотрицательное число.',self.step_prompt(form)['keyboard'])
            value=int(text)
            if field.startswith('control_') and value not in (2,3,4,5):
                return self.reply('Оценка должна быть 2, 3, 4 или 5.',self.step_prompt(form)['keyboard'])
            if field in ('classes_total','labs_assigned') and not 1<=value<=1000:
                return self.reply('Введите число от 1 до 1000.',self.step_prompt(form)['keyboard'])
            upper={'classes_attended':'classes_total','labs_submitted':'labs_assigned','late_labs':'labs_submitted'}.get(field)
            if upper and value>values[upper]:
                return self.reply('Число не может превышать '+str(values[upper])+'. Попробуйте ещё раз.',self.step_prompt(form)['keyboard'])
        values[field]=value;form['step']+=1;form['updated']=time.monotonic()
        if form['step']<len(STEPS):return self.step_prompt(form)
        start=time.perf_counter();score=float(self.tools.predict(pd.DataFrame([values]))[0]);threshold=self.tools.bundle['threshold']
        answer=(COURSES[values['course']]+f'\nСкор риска итоговой двойки: {score:.1%}. Порог: {threshold:.1%}.\n'+
            ('Есть сигнал повышенного риска.' if score>=threshold else 'Сигнал повышенного риска отсутствует.')+
            '\n\nСкор не является гарантированной вероятностью. Прогноз относится к синтетической модели и не решает вопрос об отчислении.')
        self.log('Прогноз по введённым данным: '+json.dumps(values,ensure_ascii=False),answer,'predict',(time.perf_counter()-start)*1000)
        del self.forms[chat_id];return self.reply(answer)

    def handle(self, chat_id, text):
        text=text.strip()
        aliases={'Статистика':'/stats','Прогноз':'/predict','Группа риска':'/risk','Справка':'/help',
                 'Качество модели':'/quality','Репозиторий':'/repo','Режим ответа':'/mode','Отмена':'/cancel'}
        text=aliases.get(text,text)
        if len(text)>2000:return self.reply('Вопрос слишком длинный. Максимум 2000 символов.')
        if re.search(r'\b\d{5,}:[A-Za-z0-9_-]{20,}\b',text):
            return self.reply('Похоже, это токен. Не отправляйте его в чат; настройте через файл запуска на Mac.')
        parts=text.split(maxsplit=1);command=parts[0].split('@')[0].lower() if parts else '';arg=parts[1].strip().lower() if len(parts)>1 else ''
        if command=='/cancel':
            self.forms.pop(chat_id,None);return self.reply('Ввод отменён. Выберите действие.')
        if command in ('/start','/help'):
            self.forms.pop(chat_id,None);return self.reply(HELP)
        if command=='/predict':
            if arg:
                if not re.fullmatch(r'(?:ud|bd|pr)-\d{4}',arg):return self.reply('Номер записи: ud-0001, bd-0001 или pr-0001. Для ввода своих значений отправьте /predict без номера.')
                return self.ask(chat_id,'Оцени риск для '+arg)
            self.forms[chat_id]={'step':0,'values':{},'updated':time.monotonic()};return self.step_prompt(self.forms[chat_id])
        if command in ('/stats','/risk'):
            if arg and arg not in COURSES:return self.reply('Код дисциплины: ud, bd или pr. Без кода — все дисциплины.')
            self.forms.pop(chat_id,None)
            return self.ask(chat_id,('Покажи статистику по ' if command=='/stats' else 'Покажи список группы риска по ')+arg)
        if command=='/repo':return self.reply(self.repo_fetch())
        if command=='/quality':
            m=json.loads((ARTIFACTS/'metrics.json').read_text());r=m['test'][m['selected_model']]
            return self.reply(f"Синтетический тест: {r['n']} записей.\nF1: {r['f1']:.3f}\nДоля обнаруженных двоек: {r['recall']:.1%}\nДоля верных сигналов: {r['precision']:.1%}\nROC AUC: {r['roc_auc']:.3f}\n\nКачество на реальных студентах России не проверялось.")
        if command=='/mode':
            self.forms.pop(chat_id,None)
            if arg in ('offline','local'):
                if arg=='local' and not local_available():return self.reply('Локальная LLM не подготовлена. Сейчас можно использовать обычные ответы.')
                self.modes[chat_id]=arg;return self.reply('Режим выбран: '+('обычные ответы по данным' if arg=='offline' else 'объяснения локальной LLM; сверяйте их с проверенными данными'))
            buttons=[['Обычные ответы']]
            if local_available():buttons.append(['Объяснения LLM'])
            return self.reply('Выберите режим ответа.',{'keyboard':buttons+[['Отмена']],'resize_keyboard':True})
        if text in ('Обычные ответы','Объяснения LLM'):
            return self.handle(chat_id,'/mode '+('offline' if text=='Обычные ответы' else 'local'))
        if command.startswith('/'):return self.reply('Неизвестная команда. Отправьте /help.')
        if chat_id in self.forms:return self.step(chat_id,text)
        if not text:return self.reply('Введите вопрос или выберите действие.')
        return self.ask(chat_id,text)


def configure():
    token=getpass.getpass('Вставьте токен BotFather (ввод скрыт): ').strip()
    if not valid_token(token):raise ValueError('Некорректный формат токена. Скопируйте его целиком из BotFather.')
    # Сначала подтверждаем токен у Telegram, затем сохраняем его локально.
    me=TelegramAPI(token).call('getMe')
    private_json(CONFIG,{'token':token,'mode':'offline','allowed_chat_ids':[]})
    print('Настройка сохранена локально. Бот: @'+me['username']+'. Токен не выводится.')


def run(api, controller, state_path=STATE, allowed_chat_ids=()):
    me=api.call('getMe')
    if api.call('getWebhookInfo').get('url'):
        raise ValueError('У бота уже включён webhook. Этот запуск использует polling; смену подключения выполните отдельно.')
    commands=[{'command':c,'description':d} for c,d in [('start','Начало работы'),('stats','Статистика оценок'),('predict','Прогноз риска'),('risk','Список риска'),('quality','Качество модели'),('repo','Исходный код GitHub'),('mode','Режим ответа'),('cancel','Отменить ввод'),('help','Справка')]]
    api.call('setMyCommands',{'commands':commands,'scope':{'type':'all_private_chats'}})
    state=json.loads(Path(state_path).read_text()) if Path(state_path).exists() else {}
    offset=state.get('offset',0) if state.get('bot_id')==me['id'] else 0
    allowed=set(allowed_chat_ids);pending={}
    print('Бот запущен: @'+me['username']+'. Остановка: Ctrl+C. Не закрывайте окно.')
    while True:
        try:
            delivery_uid=None
            updates=api.call('getUpdates',{'offset':offset,'timeout':25,'allowed_updates':['message']})
            for update in updates:
                uid=update['update_id'];message=update.get('message',{});chat=message.get('chat',{});chat_id=chat.get('id')
                if chat.get('type')=='private' and (not allowed or chat_id in allowed) and isinstance(message.get('text'),str):
                    # При повторе после ошибки передачи не пересчитываем ответ и шаг формы.
                    if uid not in pending:
                        try:pending[uid]=controller.handle(chat_id,message['text'])
                        except Exception:
                            controller.forms.pop(chat_id,None)
                            pending[uid]={'text':'Не удалось обработать запрос. Попробуйте /start или проверьте настройки проекта.','keyboard':KEYBOARD}
                    result=pending[uid];delivery_uid=uid
                    api.send(chat_id,result['text'],result['keyboard']);pending.pop(uid,None);delivery_uid=None
                offset=uid+1;private_json(state_path,{'bot_id':me['id'],'offset':offset})
        except TelegramError as error:
            if error.code in (401,409):raise
            if error.code==403:
                # Бот заблокирован в этом чате: пропускаем недоставленное обновление.
                if delivery_uid is not None:
                    pending.pop(delivery_uid,None);offset=delivery_uid+1;private_json(state_path,{'bot_id':me['id'],'offset':offset})
            print('Временная ошибка связи с Telegram, код '+str(error.code)+'. Повторяю подключение.')
            delay=max(3,int(error.retry_after))
            while delay>0:
                pause=min(30,delay);time.sleep(pause);delay-=pause


def main():
    parser=argparse.ArgumentParser(description='Telegram-бот учебного помощника')
    parser.add_argument('--configure',action='store_true');parser.add_argument('--configure-if-needed',action='store_true')
    args=parser.parse_args()
    try:
        if args.configure:
            configure();return
        if args.configure_if_needed and not CONFIG.exists() and not os.getenv('TELEGRAM_BOT_TOKEN'):configure()
        config=json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
        token=os.getenv('TELEGRAM_BOT_TOKEN') or config.get('token')
        if not token:raise ValueError('Токен не настроен. Запустите python -m src.telegram_bot --configure')
        if not DB_PATH.exists():raise ValueError('Сначала создайте базу: python -m src.etl')
        run(TelegramAPI(token),Controller(default_mode=config.get('mode','offline')),allowed_chat_ids=config.get('allowed_chat_ids',[]))
    except KeyboardInterrupt:print('\nБот остановлен.')
    except (ValueError,TelegramError) as error:
        print(str(error),file=sys.stderr);sys.exit(1)
    except Exception as error:
        print('Не удалось запустить бота ('+type(error).__name__+'). Проверьте базу и настройки.',file=sys.stderr);sys.exit(1)


if __name__=='__main__':main()
