# Molecular Dynamics Simulations

* Motivation

Master's computational project to learn the basic knowledge of Ising model and phase transitions.



## First steps 💡
Information to install and execute the programs.

### Pre-requisites 📋

Working environment:

```
Linux Shell and Bash
```

Sequential compilers:

```
ifort (Default)
gfortran (Must configure compile.sh)
nuitka
```

```

### Installation 🔧

The programs are ready-to-use.

Sequential program
```
compile.sh:  configure compiler and flags variables (ifort by default)

```
(computing cluster)
```
Makefile:  configure the compiler and flags variables (mpifort by default)
"sub_iqtc.sh" (1): Qsub script to run the program in the cluster with queque
"sub.sh" (2): Production runs at different temperatures
"run.sh" (1): run the program and generate de output.


```

## Execution 🚀

Sequential program
```
(1) Configure the simulation parameters (INPUT folder)
(2) Execute the "run.sh" script.
(3) Collect results in the OUTPUT folder.
    The results folder name is the date-time when the task was submitted.
(To execute binning you can use just interpreted IPython)
(In the run.sh is compiled with Nuitka, just comment these lines if you don't have the module)
Production runs
```
(1) Configure the simulation parameters input.dat in the folder production_runs/
(2) Execute the "./sub.sh input.dat" script with the ranges of temperatures scpecified by the user.
(3) Collect results in the generated folder to each production run.
(For each production run a different seed is introduced)

```
```
```
### Program-check 🔎

In the folder production-runs we have the binning and jackknife to compute for each observable statistics
In the folder results we have all the python scripts to plot the results

## Technologies 🛠️

```
- Fortran
- Open MPI subroutines
- Random numbers: mtfort
- Python
- Nuitka C compiler
- Bash shell scripts
- Computing Cluster
```

## Version 📌

Outcome : 10 / 01 / 2021 (version 1.0)

Last moified:  NONE (version --)

## Authors ✒️

* **Alexandre Sureda**
