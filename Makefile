.PHONY: lint lint-fix fmt fmt-dry typecheck override-typecheck test vtest clean

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

override-typecheck:
	basedpyright --writebaseline

test:
	python3 -m pytest

vtest:
	python3 -m pytest -vv

clean:
	rm -f out--*.png out--*.svg *.out *.err
