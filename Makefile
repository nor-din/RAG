.PHONY: all install run debug clean lint lint-strict

install:
	UV_CACHE_DIR="$$HOME/goinfre" \
	UV_PROJECT_ENVIRONMENT="$$HOME/goinfre/rag-venv" \
	UV_LINK_MODE=copy \
	uv sync

run:
	HF_HOME="$$HOME/goinfre" \
	UV_PROJECT_ENVIRONMENT="$$HOME/goinfre/rag-venv" \
	uv run python -m src search_dataset \
    	--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    	--save_directory data/output/search_results/UnansweredQuestions \
    	--k 10

debug:
	HF_HOME="$$HOME/goinfre" \
	UV_PROJECT_ENVIRONMENT="$$HOME/goinfre/rag-venv" \
	uv run python3 -m pdb -m src $(ARG)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf data/output/

lint:
	flake8 .
	mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict