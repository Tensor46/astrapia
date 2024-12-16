# version checks
.PHONY: tool-check
tool-check:
	pip install -U ruff

# code formating
.PHONY: format
format:
	ruff check --fix
	ruff format

# test
.PHONY: test
test:
	pytest -rP

# format and test
.PHONY: all
all:
	make format
	make test
