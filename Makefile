
PYTHON=~/Code/pyenvs/rocmenv/bin/python
REMOTE_DIR=/home/seb/Code/flash-attention
LOCAL_DIR=~/Code/Projects/flash-attention
PROF_FOLDER=profiling/prof_results



# run on remote
profile:
	rocprofv2 -i profiling/profile.txt -d $(PROF_FOLDER) $(PYTHON) profiling/prof_flashattn.py
	# mv results.json results.copy_stats.csv results.db results.hip_stats.csv results.stats.csv results.sysinfo.txt $(PROF_FOLDER)

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
