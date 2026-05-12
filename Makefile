PYTHON ?= python
VIDEO ?= data/raw/input_video.mp4
RUN_DIR ?= outputs/demo_room
CONFIG ?= config/default.yaml

.PHONY: setup check demo smoke test readme-assets semantics evaluate submission-assets

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

check:
	$(PYTHON) scripts/00_check_environment.py

demo: check
	$(PYTHON) scripts/run_pipeline.py --video $(VIDEO) --output $(RUN_DIR) --config $(CONFIG)
	$(PYTHON) scripts/10_make_readme_assets.py --run-dir $(RUN_DIR) --config $(CONFIG) --skip-mesh
	$(PYTHON) scripts/12_add_semantics.py --run-dir $(RUN_DIR)
	$(PYTHON) scripts/11_evaluate_reconstruction.py --run-dir $(RUN_DIR)

readme-assets:
	$(PYTHON) scripts/10_make_readme_assets.py --run-dir $(RUN_DIR) --config $(CONFIG) --skip-mesh

semantics:
	$(PYTHON) scripts/12_add_semantics.py --run-dir $(RUN_DIR)

evaluate:
	$(PYTHON) scripts/11_evaluate_reconstruction.py --run-dir $(RUN_DIR)

submission-assets:
	$(PYTHON) scripts/13_build_submission_assets.py --example-dir assets/example_outputs --assets-dir assets

smoke:
	$(PYTHON) scripts/04_estimate_depth.py --frames data/frames --output outputs/depth_smoke --depth-mode heuristic

test:
	$(PYTHON) -m pytest
