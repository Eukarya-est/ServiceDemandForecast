# Build and publish the ASP demand image to a container registry.
#
# Examples:
#   make build                                  # build local image
#   make build TAG=v0.1.0
#   make release REGISTRY=123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/asp-demand TAG=v0.1.0
#   make ecr-login REGISTRY=123456789012.dkr.ecr.ap-northeast-1.amazonaws.com AWS_REGION=ap-northeast-1

REGISTRY   ?=
IMAGE      ?= asp-demand
TAG        ?= latest
PLATFORM   ?= linux/amd64
AWS_REGION ?= ap-northeast-1

# Fully-qualified image ref: <registry>/<image>:<tag> (registry optional for local builds).
FULL_IMAGE := $(if $(REGISTRY),$(REGISTRY)/,)$(IMAGE):$(TAG)

.PHONY: build push release ecr-login run-api run-pipeline test lint

build:
	docker build --platform $(PLATFORM) -t $(FULL_IMAGE) .

push:
	docker push $(FULL_IMAGE)

release: build push ## build then push to the registry

ecr-login: ## authenticate Docker to AWS ECR
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS --password-stdin $(REGISTRY)

run-api: ## run the API container locally on :8000
	docker run --rm -p 8000:8000 \
		-v $(PWD)/data:/app/data -v $(PWD)/models:/app/models \
		$(FULL_IMAGE)

run-pipeline: ## run the Hydra pipeline in a container (pass overrides via ARGS=...)
	docker run --rm \
		-v $(PWD)/data:/app/data -v $(PWD)/models:/app/models \
		$(FULL_IMAGE) asp-demand-pipeline $(ARGS)

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run mypy asp_demand
