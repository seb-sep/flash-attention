
PYTHON=~/Code/pyenvs/rocmenv/bin/python
REMOTE_DIR=/home/seb/Code/flash-attention
LOCAL_DIR=~/Code/Projects/flash-attention
PROF_FOLDER=profiling/prof_results


BUILD_DIR=build
M_SO=$(BUILD_DIR)/m.so

.PHONY: build
build: rocm/flash_attention.hip
	rm -rf $(BUILD_DIR)/*
	MAX_JOBS=24 $(PYTHON) rocm/build.py

# Ensure m.so exists before running other commands
profile time pytorch-test profile-add: | check-m-so

check-m-so:
	@if [ ! -f $(M_SO) ]; then \
		echo "Error: $(M_SO) not found. Run 'make build' first."; \
		exit 1; \
	fi

# run on remote
profile:
	rocprofv2 -i profiling/profile.txt -d $(PROF_FOLDER) $(PYTHON) profiling/prof_flashattn.py

time:
	$(PYTHON) profiling/prof_flashattn.py

pytorch-test:
	$(PYTHON) profiling/pytorch_profile.py

profile-add:
	OUTPUT_FILE=$(PROF_FOLDER)/results.txt rocprofv2 --plugin file -i profiling/profile.txt -o foo $(PYTHON) profiling/prof_flashattn.py 
	

# run these on localhost only 
map-prof-files: 
	sshfs home-desktop:$(REMOTE_DIR)/$(PROF_FOLDER) $(LOCAL_DIR)/$(PROF_FOLDER)

unmap:
	umount $(LOCAL_DIR)/$(PROF_FOLDER)

view-prof: map-prof-files
	open -a "Brave Browser" "brave://tracing"
