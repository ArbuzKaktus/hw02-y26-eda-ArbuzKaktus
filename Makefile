.PHONY: install run

PYTHON ?= python3
VENV ?= .venv

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -r solution/requirements.txt

run:
	$(VENV)/bin/python -m streamlit run solution/app.py
