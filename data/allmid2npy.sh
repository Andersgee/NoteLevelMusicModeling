#!/bin/bash

# takes all all .mid files inside directory mid and processes
# them with a python script, which saves them as .npy inside directory npy

#ls mid/TchaikovskyPeter | while read line
ls mid | grep '\.mid' | while read line
do
	python mid2npy.py ${line:0:-4} #dont pass ending ".mid" as argument.
done
