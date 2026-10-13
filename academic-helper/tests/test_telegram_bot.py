import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
from urllib.error import URLError, HTTPError

import pytest
from src.agent import Tools
from src.config import DB_PATH
from src.telegram_bot import Controller, TelegramAPI, TelegramError, private_json, split_message, run


@pytest.fixture
def controller(tmp_path):
    path=tmp_path/'bot.sqlite';shutil.copy2(DB_PATH,path)
    return Controller(tools=Tools(db_path=path),repo_fetch=lambda:'https://github.com/azoyan28/academic-RUDN\nПоследнее изменение: test')


def test_stats_use_shared_db_and_tag_journal(controller):
    reply=controller.handle(10,'/stats ud')['text']
    assert '360' in reply and '3.16' in reply
    with sqlite3.connect(controller.tools.db_path) as db:
        assert db.execute('SELECT mode FROM requests ORDER BY request_id DESC LIMIT 1').fetchone()[0]=='telegram:offline'


def test_forecast_wizard_and_same_model(controller):
    assert 'дисциплину' in controller.handle(10,'Прогноз')['text']
    for text in ['Управление данными','3','4','16','12','4','3']:
        controller.handle(10,text)
    result=controller.handle(10,'1')['text']
    assert 'Скор риска' in result and '45.0%' in result and 10 not in controller.forms
    with sqlite3.connect(controller.tools.db_path) as db:
        assert json.loads(db.execute('SELECT tool_names FROM requests ORDER BY request_id DESC LIMIT 1').fetchone()[0])==['predict']


def test_grades_and_counts_rejected_without_advancing(controller):
    controller.handle(10,'/predict');controller.handle(10,'ud')
    assert '2, 3, 4 или 5' in controller.handle(10,'1')['text']
    assert controller.forms[10]['step']==1
    for text in ['3','4','16']:controller.handle(10,text)
    assert 'не может превышать 16' in controller.handle(10,'17')['text']
    assert controller.forms[10]['step']==4
    for text in ['12','4','2']:controller.handle(10,text)
    assert 'не может превышать 2' in controller.handle(10,'3')['text']
    assert controller.forms[10]['step']==7


def test_sessions_separate_and_cancel(controller):
    controller.handle(1,'/predict');controller.handle(2,'/predict');controller.handle(1,'ud')
    assert controller.forms[1]['step']==1 and controller.forms[2]['step']==0
    controller.handle(1,'Отмена');assert 1 not in controller.forms and 2 in controller.forms


def test_record_and_invalid_course(controller):
    assert 'ud-0001' in controller.handle(1,'/predict ud-0001')['text']
    assert 'Номер записи' in controller.handle(1,'/predict bogus')['text']
    assert 'Код дисциплины' in controller.handle(1,"/stats ud';drop table students;")['text']
    assert 'F1: 0.634' in controller.handle(1,'Качество модели')['text']
    assert 'Последнее изменение' in controller.handle(1,'Репозиторий')['text']


def test_long_and_unknown_questions(controller):
    assert 'слишком длинный' in controller.handle(1,'a'*2001)['text']
    assert 'Неизвестная команда' in controller.handle(1,'/delete')['text']
    assert 'программным генератором' in controller.handle(1,'Откуда взяты синтетические данные?')['text']


def test_unicode_messages_fit_limit():
    text='😀'*2200+'ю'*4000
    parts=split_message(text)
    assert ''.join(parts)==text
    assert all(0<len(p.encode('utf-16-le'))//2<=3500 for p in parts)
    assert split_message('**Проверенные данные**')==['Проверенные данные']


def test_private_config(tmp_path):
    path=tmp_path/'config.json';private_json(path,{'mode':'offline'})
    assert os.stat(path).st_mode & 0o777 == 0o600
    private_json(path,{'mode':'local'})
    assert json.loads(path.read_text())=={'mode':'local'}
    assert not path.with_suffix('.json.tmp').exists()


def test_transport_errors_never_print_token():
    token=str(123456)+':'+'x'*35
    def fail(request,timeout):raise URLError('Request failed: '+request.full_url)
    with pytest.raises(TelegramError) as caught:TelegramAPI(token,opener=fail).call('getMe')
    assert token not in str(caught.value) and 'api.telegram.org' not in str(caught.value)


def test_retry_after_preserved_and_methods_allowlisted():
    def fail(request,timeout):raise HTTPError(request.full_url,429,'secret',{},io.BytesIO(b'{"parameters":{"retry_after":9}}'))
    api=TelegramAPI(str(123456)+':'+'x'*35,opener=fail)
    with pytest.raises(TelegramError) as caught:api.call('getUpdates')
    assert caught.value.code==429 and caught.value.retry_after==9
    with pytest.raises(ValueError):api.call('deleteWebhook')


class FakeAPI:
    def __init__(self,updates,webhook=''):
        self.updates=updates;self.webhook=webhook;self.calls=[];self.sent=[];self.polled=False
    def call(self,name,payload=None):
        self.calls.append((name,payload))
        if name=='getMe':return {'id':10,'username':'test_bot'}
        if name=='getWebhookInfo':return {'url':self.webhook}
        if name=='setMyCommands':return True
        if name=='getUpdates':
            if self.polled:raise KeyboardInterrupt
            self.polled=True;return self.updates
    def send(self,chat,text,keyboard):self.sent.append((chat,text))


def test_polling_private_only_and_persistent_offset(controller,tmp_path):
    updates=[{'update_id':20,'message':{'chat':{'id':2,'type':'group'},'text':'/stats'}},
             {'update_id':21,'message':{'chat':{'id':1,'type':'private'},'text':'/stats ud'}}]
    api=FakeAPI(updates);path=tmp_path/'state.json'
    with pytest.raises(KeyboardInterrupt):run(api,controller,state_path=path)
    assert len(api.sent)==1 and api.sent[0][0]==1
    assert json.loads(path.read_text())['offset']==22
    assert api.calls[-1]==('getUpdates',{'offset':22,'timeout':25,'allowed_updates':['message']})


def test_allowed_chats_and_old_bot_cursor(controller,tmp_path):
    path=tmp_path/'state.json';private_json(path,{'bot_id':999,'offset':500})
    api=FakeAPI([{'update_id':1,'message':{'chat':{'id':8,'type':'private'},'text':'/stats'}}])
    with pytest.raises(KeyboardInterrupt):run(api,controller,state_path=path,allowed_chat_ids=[1])
    assert api.sent==[]
    assert next(payload for name,payload in api.calls if name=='getUpdates')['offset']==0


def test_existing_webhook_not_removed(controller,tmp_path):
    api=FakeAPI([],webhook='https://example.test/hook')
    with pytest.raises(ValueError,match='webhook'):run(api,controller,state_path=tmp_path/'state.json')
    assert not any(name=='deleteWebhook' for name,_ in api.calls)


def test_token_message_not_echoed_or_logged(controller):
    token=str(123456)+':'+'x'*35
    with sqlite3.connect(controller.tools.db_path) as db:before=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
    answer=controller.handle(1,token)['text']
    assert token not in answer and 'токен' in answer
    with sqlite3.connect(controller.tools.db_path) as db:assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==before


def test_failed_delivery_reuses_answer(controller,tmp_path,monkeypatch):
    import src.telegram_bot as module
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    update={'update_id':4,'message':{'chat':{'id':1,'type':'private'},'text':'/stats ud'}}
    class RetryAPI(FakeAPI):
        def __init__(self):super().__init__([update]);self.round=0;self.fail=True
        def call(self,name,payload=None):
            if name=='getUpdates':
                self.round+=1
                if self.round>2:raise KeyboardInterrupt
                return [update]
            return super().call(name,payload)
        def send(self,chat,text,keyboard):
            if self.fail:self.fail=False;raise TelegramError()
            super().send(chat,text,keyboard)
    api=RetryAPI();calls=[];original=controller.handle
    def counted(chat,text):calls.append(text);return original(chat,text)
    monkeypatch.setattr(controller,'handle',counted)
    with pytest.raises(KeyboardInterrupt):run(api,controller,state_path=tmp_path/'state.json')
    assert calls==['/stats ud'] and len(api.sent)==1


def test_bad_request_does_not_stop_polling(controller,tmp_path,monkeypatch):
    def fail(*args):raise RuntimeError('internal-sensitive-error')
    monkeypatch.setattr(controller,'handle',fail)
    api=FakeAPI([{'update_id':7,'message':{'chat':{'id':1,'type':'private'},'text':'/stats'}}])
    with pytest.raises(KeyboardInterrupt):run(api,controller,state_path=tmp_path/'state.json')
    assert 'Не удалось обработать' in api.sent[0][1] and 'internal-sensitive-error' not in api.sent[0][1]
    assert json.loads((tmp_path/'state.json').read_text())['offset']==8
