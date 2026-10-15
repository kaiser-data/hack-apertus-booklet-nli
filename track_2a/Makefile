.PHONY: run build cases predict batch test lint report modal-deploy modal-smoke

IMAGE ?= booklet-nli
MODE  ?= rag
K     ?= 16
SPLIT ?= test
LIMIT ?= 60
RUN   ?= runs/$(SPLIT)-$(MODE)-k$(K)-n$(LIMIT)

# Needs LLM_NAME, LLM_BASE_URL and LLM_API_KEY in the environment (see README).
# For a server on the host (e.g. Ollama) use LLM_BASE_URL=http://host.docker.internal:11434/v1
DOCKER_ENV = -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY -e BASE_URL -e API_KEY \
	--add-host=host.docker.internal:host-gateway
DOCKER_RUN = docker run --rm $(DOCKER_ENV) -e BOOKLET_NLI_DATA=/app/data \
	-v $(CURDIR)/data:/app/data -v $(CURDIR)/runs:/app/runs --entrypoint /app/.venv/bin/booklet-nli $(IMAGE)

# Default for judges: evaluate on a balanced sample of the held-out split and print metrics.
run: build
	@mkdir -p data runs
	$(DOCKER_RUN) eval --split $(SPLIT) --limit $(LIMIT) --mode $(MODE) --k $(K) --out /app/$(RUN)

build:
	docker build --platform linux/amd64 -t $(IMAGE) .

# The official evaluation contract (the image's default entrypoint):
# make cases CASES=<dir with cases.jsonl and booklets/> OUT=<dir for predictions.jsonl>
CASES ?= examples
OUT   ?= runs/cases-out
cases: build
	@mkdir -p $(OUT)
	docker run --rm --platform linux/amd64 $(DOCKER_ENV) \
		-v $(abspath $(CASES)):/data:ro -v $(abspath $(OUT)):/output $(IMAGE) \
		--input /data/cases.jsonl --output /output/predictions.jsonl

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

# The technical report: figures from the committed run files, then LaTeX (Tectonic).
report:
	uv run --with matplotlib python paper/figures.py
	uv run --with matplotlib python paper/pipeline_diagram.py
	cd paper && tectonic -X compile report.tex --outdir .
	cp paper/report.pdf kaiser-data_Report.pdf
