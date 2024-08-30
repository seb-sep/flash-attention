
PYTHON=~/Code/pyenvs/rocmenv/bin/python
REMOTE_DIR=/home/seb/Code/flash-attention
LOCAL_DIR=~/Code/Projects/flash-attention
PROF_FOLDER=profiling/prof_results



# run on remote
profile:
	rocprof -i profiling/profile.txt -t /tmp -d $(PROF_FOLDER) $(PYTHON) profiling/prof_flashattn.py
	mv results.json results.copy_stats.csv results.db results.hip_stats.csv results.stats.csv results.sysinfo.txt $(PROF_FOLDER)

profile-add:
	rocprof --tool-version 1 --hip-trace -d $(PROF_FOLDER) $(PYTHON) profiling/vecadd.py
	mv results.json results.copy_stats.csv results.db results.hip_stats.csv results.stats.csv results.sysinfo.txt $(PROF_FOLDER)

# run these on localhost only 
map-prof-files: 
	sshfs home-desktop:$(REMOTE_DIR)/$(PROF_FOLDER) $(LOCAL_DIR)/$(PROF_FOLDER)

unmap:
	umount $(LOCAL_DIR)/$(PROF_FOLDER)

view-prof: map-prof-files
	open -a "Brave Browser" "brave://tracing"
