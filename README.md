# fep
This is a fully functional part of the free energy prediction code that allows for generating soap descriptors, optimization of the model parameters of prediction.
If used, please cite: https://www.nature.com/articles/s41524-021-00638-x


## Installation
The framework requires standard python3 release (like Anaconda with numpy, etc) as well as Dscribe (https://github.com/SINGROUP/dscribe) for the computation of the SOAP descriptor. 


## Operation

### 1. Descriptor creation
To create soap descriptors put relaxed strures files in "\training" and "\prediction" folders, each structure in a appropriate sub folder named after the name of the structure (for example: "\training\butane\" or "\prediction\ice\"). Structure files has to be named: "geometry.in.next_step" (for FHI-AIMS) or "lmp.data.relax" (for lammps). Next, run from the "\fep" directory:
```
python run_desc_generator.py
```
This will creat "soap.npz" file containing SOAP descriptors as well as POSCAR files with structure data in appropriate sub folders.

### 2. Hype-parameter optimization
This script, apart form the descriptors in the "\training\" folder, requires additional file, called "fe.dat" containing the free energy data. The "fe.dat" file has the first line being a header, next, two columns: first containing names of the training structures (that has to be the same as the name of appropriate sub-folders), second column contains free energy per structure. To run the optimization run from the "fep" directory:
```
run_hyperparam_opt.py
```
This will create "hp.npz" file in the "\training\" directory, containing hyper parameters.

### 3. Prediction
To perform prediction of the free energy run from the "\fep" directory:
```
run_prediction.py
```
This will create a "fep.dat" file in the "\prediction\" sub-folder. The file contains predicted free energy values and is formatted the same way as the "fe.dat" file

### 4. Prediction analysis
To display obtained results and compare them (if available) with "fe.data" of the "\prediction\" structures, run from the "fep" directory:
```
run_prediction_analysis.py
```
This will display the predicted free energy next to the benchmark data.
