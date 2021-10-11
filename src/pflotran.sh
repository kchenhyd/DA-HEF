#!/bin/bash -l
nreaz=$1
ncore=$2
mpirun=$3
exeprg=$4

name="1dthermal"
dir="pflotran_results/"

cd $dir
$mpirun -n $ncore $exeprg -pflotranin $name".in" -stochastic -num_realizations $nreaz -num_groups $ncore -screen_output off ;
wait
cd ../
