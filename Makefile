.PHONY: data test lint
data:
	python download_data.py
test:
	python -m unittest discover -s tests -v
lint:
	python -m compileall -q .
