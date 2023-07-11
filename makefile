PROJECT_NAME=craterslab

formatcheck:
	isort $(PROJECT_NAME)
	black  .
	
typecheck:
	mypy .

lint:
	flake8 .

test:
	pytest

qa: formatcheck lint test typecheck  
