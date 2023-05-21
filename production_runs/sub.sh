#!/bin/bash

targetFile=$1
temp=$1
if [ -f  ./$targetFile ]; then
        echo 'Beguinning the production runs'
        echo 'Tinit  = 2.0'
        echo 'Tfin   = 3.0'
        echo 'deltaT = 0.1'
else
        echo 'The target input file does not exist:'
        echo 'Please, select a valid one.'
        echo 'EXITING PROGRAM'
        exit
fi
cd ..
cd PROGRAM/

cp r_main ../

cd ..
mv r_main production_runs
cd production_runs

T="2.0"
while [ $T != "3.1" ]; do
        echo 'BEGUINNING PRDUCTION RUN AT T = '$T
	
        mkdir 'productionn'$T
        cp r_main 'productionn'$T/
        cd 'productionn'$T
        cp ../$targetFile $temp
        sed -i 's/Temp/'$T'/g' $temp
        ./r_main $temp

        cd ..
        echo 'RUN FINISHED'

        T=$(echo $T" + 0.1" | bc)
done

echo 'EXITING PROGRAM: EVERYTHING CORRECT!'


