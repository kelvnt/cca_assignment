activate:
	uv venv
	(source .venv/bin/activate)

install:
	uv sync
	uv run pre-commit install

init_project:
	git init
	uv sync
	uv run pre-commit install

python_src:
	export PYTHONPATH=$(PWD)

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
