uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

set-dvc:
	dvc remote add origin s3://dvc
	dvc remote modify origin endpointurl https://dagshub.com/ssime-git/DVC_within_docker.s3
	dvc remote modify origin --local access_key_id XXX
	dvc remote modify origin --local secret_access_key XXX

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

track:
	git add .
	git commit -m "updating the run"
	dvc commit
	git push
	dvc push

dvc-docto:
	docker compose run --rm dvc-runner dvc doctor

all: init-git init-dvc build run