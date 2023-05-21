#!/bin/bash
#!/usr/bin/env python
#Getting the date and time to create the
tmp_folder=$(date +'%Y_%m_%d_%H_%M_%S')
#echo $(date)
#echo $tmp_folder

# Creating the temporal file in the PROGRAM dir.

mkdir PROGRAM/$tmp_folder/
#mkdir OUTPUT/

# Copy the INPUT filles to tmp. folder
cp INPUT/* PROGRAM/$tmp_folder/

# Execute compiler to create or actualize programs
cd PROGRAM/
./compile.sh
#gprof created

#Copy the program scripts in the tmp. folder

cp *.py $tmp_folder/
#cp *.bin $tmp_folder/
cp r_* $tmp_folder/



# Running the programs in the tmp folder

cd $tmp_folder/

./r_main input.dat
nuitka  --nofollow-imports binning.py
./binning.bin
#python3 binning.py




# After the program ends, create tar, delete folder and move to OUTPUT
cd ..
cd ..
pwd
mkdir OUTPUT/$tmp_folder/
mkdir OUTPUT/$tmp_folder/figures/
mkdir OUTPUT/$tmp_folder/data/
mv PROGRAM/$tmp_folder/*.png OUTPUT/$tmp_folder/figures/
mv PROGRAM/$tmp_folder/*.dat OUTPUT/$tmp_folder/data/
mv PROGRAM/$tmp_folder/* OUTPUT/$tmp_folder/


rm -r PROGRAM/$tmp_folder/

echo "Folder for this simulation ${tmp_folder}"