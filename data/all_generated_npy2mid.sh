#!/bin/bash

# takes all all .npy files inside directory generated_npy and processes
# them with a python script, which saves them as .mid inside directory generated_mid

ls generated_npy | grep '\.npy' | while read line
do
	python generated_npy2mid.py ${line:0:-4} #dont pass ending ".npy" as argument.
done
