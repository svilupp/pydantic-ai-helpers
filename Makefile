.PHONY: help install install-dev format lint type test test-watch coverage docs clean build release

# Default target
.DEFAULT_GOAL := help

# Python executable
PYTHON := python3

# Colors for output
COLOR_RESET = \033[0m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
COLOR_BLUE = \033[34m

help: ## Show this help message
	@echo "$(COLOR_BLUE)pydantic-ai-utils Development Commands$(COLOR_RESET)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-15s$(COLOR_RESET) %s\n", $$1, $$2}'

install: ## Install package in production mode
	@echo "$(COLOR_YELLOW)Installing pydantic-ai-utils...$(COLOR_RESET)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install .
	@echo "$(COLOR_GREEN)✓ Installation complete$(COLOR_RESET)"

install-dev: ## Install package in development mode with all dependencies
	@echo "$(COLOR_YELLOW)Installing pydantic-ai-utils in development mode...$(COLOR_RESET)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install
	@echo "$(COLOR_GREEN)✓ Development installation complete$(COLOR_RESET)"

format: ## Format code with ruff
	@echo "$(COLOR_YELLOW)Formatting code...$(COLOR_RESET)"
	ruff format src tests docs
	ruff check --fix src tests
	@echo "$(COLOR_GREEN)✓ Formatting complete$(COLOR_RESET)"

lint: ## Run all linting checks
	@echo "$(COLOR_YELLOW)Running linting checks...$(COLOR_RESET)"
	@echo "Running ruff..."
	ruff check src tests
	@echo "Running ruff format check..."
	ruff format --check src tests docs
	@echo "$(COLOR_GREEN)✓ Linting complete$(COLOR_RESET)"

type: ## Run type checking with mypy
	@echo "$(COLOR_YELLOW)Running type checks...$(COLOR_RESET)"
	mypy src tests
	@echo "$(COLOR_GREEN)✓ Type checking complete$(COLOR_RESET)"

test: ## Run tests with coverage
	@echo "$(COLOR_YELLOW)Running tests...$(COLOR_RESET)"
	pytest tests/ -v --cov=pydantic_ai_utils --cov-report=term-missing --cov-report=html
	@echo "$(COLOR_GREEN)✓ Tests complete$(COLOR_RESET)"
	@echo "Coverage report available at htmlcov/index.html"

test-watch: ## Run tests in watch mode
	@echo "$(COLOR_YELLOW)Running tests in watch mode...$(COLOR_RESET)"
	@echo "Install pytest-watch first: pip install pytest-watch"
	ptw tests/ -- -v

coverage: ## Generate and open coverage report
	@echo "$(COLOR_YELLOW)Generating coverage report...$(COLOR_RESET)"
	pytest tests/ --cov=pydantic_ai_utils --cov-report=html --quiet
	@echo "$(COLOR_GREEN)✓ Coverage report generated$(COLOR_RESET)"
	@echo "Opening coverage report..."
	@$(PYTHON) -m webbrowser htmlcov/index.html || open htmlcov/index.html || echo "Please open htmlcov/index.html manually"

docs: ## Build and serve documentation
	@echo "$(COLOR_YELLOW)Building documentation...$(COLOR_RESET)"
	@echo "Note: Documentation building not yet configured"
	@echo "TODO: Add sphinx or mkdocs configuration"

clean: ## Remove build artifacts and cache files
	@echo "$(COLOR_YELLOW)Cleaning build artifacts...$(COLOR_RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "$(COLOR_GREEN)✓ Cleanup complete$(COLOR_RESET)"

build: clean ## Build distribution packages
	@echo "$(COLOR_YELLOW)Building distribution packages...$(COLOR_RESET)"
	$(PYTHON) -m pip install --upgrade build twine
	$(PYTHON) -m build
	@echo "$(COLOR_GREEN)✓ Build complete$(COLOR_RESET)"
	@echo "Checking package..."
	twine check dist/*

release: ## Create a new release (use: make release VERSION=0.1.0)
	@if [ -z "$(VERSION)" ]; then \
		echo "$(COLOR_YELLOW)Error: VERSION not specified$(COLOR_RESET)"; \
		echo "Usage: make release VERSION=0.1.0"; \
		exit 1; \
	fi
	@echo "$(COLOR_YELLOW)Creating release v$(VERSION)...$(COLOR_RESET)"
	@echo "Updating version in pyproject.toml..."
	@sed -i.bak 's/version = ".*"/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak
	@sed -i.bak 's/__version__ = ".*"/__version__ = "$(VERSION)"/' src/pydantic_ai_utils/__init__.py && rm src/pydantic_ai_utils/__init__.py.bak
	@echo "Updating CHANGELOG.md..."
	@sed -i.bak "s/\[Unreleased\]/[$(VERSION)] - $$(date +%Y-%m-%d)/" CHANGELOG.md && rm CHANGELOG.md.bak
	@echo "Committing changes..."
	git add pyproject.toml src/pydantic_ai_utils/__init__.py CHANGELOG.md
	git commit -m "Release v$(VERSION)"
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@echo "$(COLOR_GREEN)✓ Release v$(VERSION) created$(COLOR_RESET)"
	@echo "Push with: git push origin main --tags"

check: lint type test ## Run all checks (lint, type, test)
	@echo "$(COLOR_GREEN)✓ All checks passed!$(COLOR_RESET)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(COLOR_YELLOW)Running pre-commit hooks...$(COLOR_RESET)"
	pre-commit run --all-files
	@echo "$(COLOR_GREEN)✓ Pre-commit checks complete$(COLOR_RESET)"

dev: install-dev ## Alias for install-dev
	@echo "$(COLOR_GREEN)✓ Ready for development!$(COLOR_RESET)"