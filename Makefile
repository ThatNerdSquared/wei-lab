.PHONY: run lint fmt

# SRCS = $(shell find ./recombinator -type f -name '*.py')

run:
	python3 ./main.py

# lint:
# 	python3 -m flake8 $(SRCS)

# fmt:
# 	python3 -m black $(SRCS)
