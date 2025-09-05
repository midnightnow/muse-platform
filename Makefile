.PHONY: run dev build docker-build docker-run smoke clean check-once
PORT ?= 9000

run:        ## Start agent server locally
	uvicorn agent.main:app --host 0.0.0.0 --port $(PORT)

dev:        ## Install dependencies and start server
	python -m pip install -r requirements.txt && $(MAKE) run

smoke:      ## Verify health endpoint returns ok
	curl -fsS http://127.0.0.1:$(PORT)/health | jq .

check-once: ## Run health check once and exit (CI/CD gate)
	python3 -m agent.main --once

docker-build: ## Build container image
	docker build -t agent-001 .

docker-run: ## Run container with health checks
	docker run -d --rm -p $(PORT):$(PORT) --name agent-001 \
		-e PORT=$(PORT) -e TARGETS="http://localhost:8000/health" agent-001

ecosystem: ## Deploy complete ecosystem
	./ecosystem-deploy.sh

clean:     ## Clean up containers
	docker rm -f agent-001 2>/dev/null || true

help:      ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'