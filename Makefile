.PHONY: lint lint-fix fmt fmt-dry typecheck test vtest clean

all: fmt-dry lint typecheck test

lint:
	ruff check

lint-fix:
	ruff check --fix

fmt:
	ruff format

fmt-dry:
	ruff format --check

typecheck:
	basedpyright

test:
	python3 -m pytest

vtest:
	python3 -m pytest -vv

clean:
	rm -f out--*.png
	rm -f **/out--*.png
