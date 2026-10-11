.PHONY: run build predict batch test lint modal-deploy modal-smoke

IMAGE ?= booklet-nli
MODE  ?= rag
K     ?= 8
SPLIT ?= test
LIMIT ?= 60
RUN   ?= runs/$(SPLIT)-$(MODE)-k$(K)-n$(LIMIT)

# Needs LLM_NAME, LLM_BASE_URL and LLM_API_KEY in the environment (see README).
# For a server on the host (e.g. Ollama) use LLM_BASE_URL=http://host.docker.internal:11434/v1
DOCKER_RUN = docker run --rm -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY \
	--add-host=host.docker.internal:host-gateway \
	-v $(CURDIR)/data:/app/data -v $(CURDIR)/runs:/app/runs $(IMAGE)

# Default for judges: evaluate on a balanced sample of the held-out split and print metrics.
run: build
	@mkdir -p data runs
	$(DOCKER_RUN) eval --split $(SPLIT) --limit $(LIMIT) --mode $(MODE) --k $(K) --out /app/$(RUN)

build:
	docker build -t $(IMAGE) .

# make predict BOOKLET=<pdf url> CLAIM="..."
predict: build
	$(DOCKER_RUN) predict --booklet "$(BOOKLET)" --claim "$(CLAIM)" --mode $(MODE) --k $(K)

# make batch INPUT=data/claims.jsonl OUTPUT=runs/predictions.jsonl  (paths inside track_2a/)
batch: build
	$(DOCKER_RUN) batch --input /app/$(INPUT) --output /app/$(OUTPUT) --mode $(MODE) --k $(K)

test:
	uv run pytest -q

lint:
	uv run ruff check src tests modal

modal-deploy:
	modal deploy modal/serve_apertus.py

modal-smoke:
	modal run modal/serve_apertus.py
