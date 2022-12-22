VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.venv: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

setup: .venv

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

train:
	$(PYTHON) ./main/train.py

run:
	$(PYTHON) ./main/predict.py

start: setup train run

.DEFAULT_GOAL = start
