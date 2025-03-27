uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	
init-git:
	git init
	git add .gitignore Dockerfile docker-compose.yml dvc.yaml src/
	git commit -m "Initial project structure and pipeline definition"

init-dvc:
	dvc init
	git add .dvc/config .dvc/.gitignore # Add DVC config files
	git commit -m "Initialize DVC"

build:
	docker compose build

run:
	docker compose run --rm dvc-runner dvc repro

all: init-git init-dvc build run