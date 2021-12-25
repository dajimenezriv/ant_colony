#!/bin/bash
#
# SERVER
# eowyn.ac.tuwien.ac.at
# 
# SEND FILE FROM CLUSTER TO LOCAL
# scp hot12119846@eowyn.ac.tuwien.ac.at:/home1/hot12119846/file .
#
# SEND FOLDER FROM LOCAL TO CLUSTER
# scp -r . hot12119846@eowyn.ac.tuwien.ac.at:/home1/hot12119846/
#
# SEND ALL FILES FROM LOCAL DIR THAT DO NOT START WITH DOT
# scp -r [!.]* hot12119846@eowyn.ac.tuwien.ac.at:/home1/hot12119846/
#
# qsub -N dajiri -l h_vmem=2G -r y -e /dev/null -o /dev/null run.sh
# qsub -N dajiri -l h_vmem=2G -r y run.sh
# qstat
#

/usr/bin/python3 ./src/main.py > output.txt
