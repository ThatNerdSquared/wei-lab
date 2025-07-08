.PHONY: lint lint-fix fmt fmt-dry typecheck clean

all: lint fmt-dry typecheck

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

clean:
	rm -f out--*.png
	rm -f **/out--*.png
