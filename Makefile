# Remove compiled bytecode of source files
dev-clean-pyc:
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
	@find . -name '*.pytest_cache' -exec rm -fr {} +

# Setup the project environment by:
# - Run pipenv shell to start the virtual env
env:
	pipenv --python=$(conda run which python) --site-packages
	pipenv shell

# Install all libraries of package
install-all:
	pipenv install --system --dev

# Install a package 
install:
	pipenv install $(pkg)

# Install a package in dev mode
install-dev:
	pipenv install --dev $(pkg)

# Run the model
model:
	pipenv run python src/main.py

# Run tensorboard
board:
	tensorboard --logdir storage/logs/fit-1711494431 

# Run the model and predict a random value from our dataset.
predict:
	pipenv run python src/predict.py

# Run the scrapper to create or improve our dataset.
scrapper:
	pipenv run python src/scrapper.py

# Run the dataset cleaner to create the clean_dataset
cleaner:
	pipenv run python src/cleaner.py
