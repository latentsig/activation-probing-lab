.PHONY: install install-train test smoke data

install:
	python -m pip install -e .

install-train:
	python -m pip install -e '.[train]'

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python -m activation_probing_lab --config configs/qwen3-4b-toy.yaml smoke-demo

data:
	PYTHONPATH=src python -m activation_probing_lab --config configs/qwen3-4b-toy.yaml generate-data
