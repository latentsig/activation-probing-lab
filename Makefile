.PHONY: install install-train install-mlx test smoke data pilot-mlx

install:
	python -m pip install -e .

install-train:
	python -m pip install -e '.[train]'

install-mlx:
	python -m pip install -e '.[mlx]'

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python -m activation_probing_lab --config configs/qwen3-4b-toy.yaml smoke-demo

data:
	PYTHONPATH=src python -m activation_probing_lab --config configs/qwen3-4b-toy.yaml generate-data

pilot-mlx:
	PYTHONPATH=src python -m activation_probing_lab --config configs/qwen3.5-4b-mlx-pilot.yaml generate-data
	PYTHONPATH=src python -m activation_probing_lab --config configs/qwen3.5-4b-mlx-pilot.yaml train
	PYTHONPATH=src python -m activation_probing_lab --config configs/qwen3.5-4b-mlx-pilot.yaml capture
