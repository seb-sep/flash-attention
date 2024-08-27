
PYTHON=~/Code/pyenvs/rocmenv/bin/python

profile:
	rocprof --tool-version 2 --hip-trace $(PYTHON) profiling/prof_flashattn.py   