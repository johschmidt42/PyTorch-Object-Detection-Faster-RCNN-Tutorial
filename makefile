SHELL=/bin/bash

# MAKEFILE to quickly execute utility commands
#
# You can use tab for autocomplete in your terminal
# > make[space][tab]
#

format:
	@echo "Running isort..."
	@echo $(shell pwd)
	@isort $(shell pwd) --profile "black"
	@echo "Running black..."
	@black $(shell pwd)

check-format:
	@echo "Running check isort..."
	@isort --check $(shell pwd) || true
	@echo "Running check flake8..."
	@flake8 $(shell pwd) --ignore=E501 || true
