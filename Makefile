
PYTHON=~/Code/pyenvs/rocmenv/bin/python
REMOTE_DIR=/home/seb/Code/flash-attention
LOCAL_DIR=~/Code/Projects/flash-attention



profile:
	rocprof --tool-version 2 --hip-trace $(PYTHON) profiling/prof_flashattn.py   

# run on localhost only 
map-prof-files: 
	sshfs home-desktop:$(REMOTE_DIR)/profiling $(LOCAL_DIR)/profiling

unmap:
	umount $(LOCAL_DIR)/profiling

view-prof: map-prof-files
	open -a "Brave Browser" "brave://tracing"
