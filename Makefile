.PHONY: clean

# SRCS = $(shell find ./recombinator -type f -name '*.py')

clean:
	rm -f out--*.png
	rm -f **/out--*.png

# lint:
# 	python3 -m flake8 $(SRCS)

# fmt:
# 	python3 -m black $(SRCS)
