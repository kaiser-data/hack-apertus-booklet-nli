# The project lives in track_2a/; this forwards every target there, e.g. `make run LIMIT=976 RUN=runs/test-fresh`.
.PHONY: run build predict batch test lint
run build predict batch test lint:
	$(MAKE) -C track_2a $@
