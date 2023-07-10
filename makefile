PROJECT_NAME=craterslab

formatcheck:
	isort -rc $(PROJECT_NAME)
	black --check .
	
typecheck:
	mypy

lint:
	flake8 

test:
	pytest

qa: formatcheck typecheck lint test
