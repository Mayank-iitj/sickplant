# Makefile for Plant Disease Detector

.PHONY: help install install-dev install-prod test lint format clean docker-build docker-run docker-push deploy-compose deploy-k8s

# Configuration
PYTHON := python
PIP := pip
DOCKER_REGISTRY := docker.io/yourname
IMAGE_NAME := plant-disease-detector
VERSION := latest

help: ## Show this help message
	@echo "Plant Disease Detector - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black isort flake8 mypy

install-prod: ## Install production dependencies
	$(PIP) install -r requirements-prod.txt

# Testing
test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v

# Code Quality
lint: ## Run linters
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/
	isort --check-only src/ tests/

# Dataset
dataset-dummy: ## Create dummy dataset
	$(PYTHON) examples/download_dataset.py --dummy

dataset-download: ## Download PlantVillage dataset
	$(PYTHON) examples/download_dataset.py --output data/plantvillage

# Training
train-demo: ## Train demo model on dummy dataset
	$(PYTHON) src/cli.py train \
		--data data/dummy_dataset \
		--output models/demo \
		--model resnet18 \
		--epochs 5 \
		--batch 8 \
		--lr 0.001

train-full: ## Train full model
	$(PYTHON) src/cli.py train \
		--data data/plantvillage \
		--output models/production \
		--model efficientnet_b2 \
		--epochs 50 \
		--batch 32 \
		--lr 0.001

# Evaluation
evaluate: ## Evaluate trained model
	$(PYTHON) src/cli.py evaluate \
		--model models/demo/best_model.pth \
		--data data/dummy_dataset/test

# Inference
predict: ## Run prediction on sample image
	$(PYTHON) src/cli.py predict \
		--model models/demo/best_model.pth \
		--image data/dummy_dataset/test/healthy/healthy_test_001.jpg \
		--explain

# Serving
serve-streamlit: ## Start Streamlit web UI
	$(PYTHON) src/cli.py serve \
		--model models/demo/best_model.pth \
		--port 8501

serve-api: ## Start FastAPI server
	uvicorn src.serve.app_api:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

# Docker
docker-build: ## Build all Docker images
	docker build --target cpu -t $(IMAGE_NAME):$(VERSION) .
	docker build --target api -t $(IMAGE_NAME):api-$(VERSION) .

docker-build-gpu: ## Build GPU Docker image
	docker build --target gpu -t $(IMAGE_NAME):gpu-$(VERSION) .

docker-run: ## Run Docker container (CPU)
	docker run -p 8501:8501 \
		-v $$(pwd)/models:/app/models:ro \
		$(IMAGE_NAME):$(VERSION)

docker-run-api: ## Run API Docker container
	docker run -p 8000:8000 \
		-v $$(pwd)/models:/app/models:ro \
		-e MODEL_PATH=/app/models/demo/best_model.pth \
		$(IMAGE_NAME):api-$(VERSION)

docker-push: ## Push Docker images to registry
	docker tag $(IMAGE_NAME):$(VERSION) $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker tag $(IMAGE_NAME):api-$(VERSION) $(DOCKER_REGISTRY)/$(IMAGE_NAME):api-$(VERSION)
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):api-$(VERSION)

# Docker Compose
compose-up: ## Start services with Docker Compose
	docker-compose up -d

compose-down: ## Stop Docker Compose services
	docker-compose down

compose-logs: ## View Docker Compose logs
	docker-compose logs -f

# Kubernetes
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f kubernetes/deployment.yaml

k8s-status: ## Check Kubernetes deployment status
	kubectl get all -n plant-disease-detector

k8s-logs: ## View Kubernetes logs
	kubectl logs -f deployment/plant-disease-api -n plant-disease-detector

k8s-delete: ## Delete Kubernetes deployment
	kubectl delete -f kubernetes/deployment.yaml

# Cleanup
clean: ## Clean up temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage

clean-data: ## Clean data directories (WARNING: deletes data!)
	rm -rf data/dummy_dataset data/plantvillage

clean-models: ## Clean model directories (WARNING: deletes models!)
	rm -rf models/*

clean-all: clean clean-data clean-models ## Clean everything

# Monitoring
tensorboard: ## Start TensorBoard
	tensorboard --logdir models/demo/tensorboard --port 6006

# Deployment
deploy-local: docker-build compose-up ## Build and deploy locally

deploy-production: docker-build docker-push k8s-deploy ## Deploy to production (Kubernetes)

# Quick start
quickstart: install dataset-dummy train-demo serve-streamlit ## Complete quickstart workflow

# Verification
verify: ## Verify installation
	$(PYTHON) verify_installation.py

# Documentation
docs: ## Build documentation (placeholder)
	@echo "Documentation build not implemented yet"

# Environment
env-create: ## Create virtual environment
	$(PYTHON) -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

env-export: ## Export current environment
	$(PIP) freeze > requirements-freeze.txt

# === Streamlit Deployment Commands ===

.PHONY: streamlit-local streamlit-test streamlit-check streamlit-setup

streamlit-local: ## Run Streamlit app locally
streamlit run streamlit_app.py

streamlit-test: ## Test Streamlit deployment locally
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py

streamlit-check: ## Check Streamlit deployment readiness
@echo "Checking Streamlit deployment readiness..."
@test -f streamlit_app.py && echo "✓ streamlit_app.py exists" || echo "✗ streamlit_app.py missing"
@test -f requirements-streamlit.txt && echo "✓ requirements-streamlit.txt exists" || echo "✗ requirements-streamlit.txt missing"
@test -f packages.txt && echo "✓ packages.txt exists" || echo "✗ packages.txt missing"
@test -f .streamlit/config.toml && echo "✓ .streamlit/config.toml exists" || echo "✗ config.toml missing"
@test -f .gitattributes && echo "✓ .gitattributes exists" || echo "✗ .gitattributes missing"
@echo "Ready for Streamlit Cloud deployment!"

streamlit-setup: ## Setup Streamlit secrets locally
@mkdir -p .streamlit
@cp .streamlit/secrets.toml.example .streamlit/secrets.toml 2>/dev/null || cp .streamlit/secrets.toml.template .streamlit/secrets.toml
@echo "Edit .streamlit/secrets.toml with your configuration"

git-lfs-setup: ## Setup Git LFS for model files
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
@echo "Git LFS configured for model files"
