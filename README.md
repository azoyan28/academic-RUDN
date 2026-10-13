# Учебный помощник по успеваемости

Россия, оценки 2–5. Учебный проект Азояна Ивана Степановича. Все записи синтетические; вуз вымышленный.

Код и подробные инструкции: [academic-helper](academic-helper/README.md).

```bash
git clone https://github.com/azoyan28/academic-RUDN.git
cd academic-RUDN/academic-helper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.etl
streamlit run app.py --server.address 127.0.0.1
```

Локальная LLM необязательна для автономного режима. Загрузка её весов описана в основной инструкции.
