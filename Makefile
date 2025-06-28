.PHONY: clean

clean:
	rm -f out--*.png
	rm -f **/out--*.png

lint:
	ruff check

lint-fix:
	ruff check --fix

fmt:
	ruff format

fmt-dry:
	ruff format --check
