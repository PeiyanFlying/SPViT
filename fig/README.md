# AQFP Placement and Routing Tool

# Compile

## System Requirement

+ -std=c++17
+ qt 
+ g++9

## Build through CMake

```bash
~$ git clone https://github.com/Oilivia-C/AQFP_P-R_tool.git
~$ mkdir AQFP_P-R_tool/build
~$ cd build
~$ cmake ../
~$ make 
```

# Run Qt to draw the result
Under bin folder, run:
```bash
~$ ./qt_aqfp <design_name>.png
```


# Run Testcase
Before running, set the .params file under testcases folder as following section.
Under bin folder, run:
```bash
~$ ./main ../testcase/<design_name>/<design_name>.params
```

## Parameter Settings

### <design_name>.params
| Parameter | Description | Example |
| --------- | ----------- | ------- |
| design name | name of the circuit | neuron_maj |
| cell_library | A library file with information of all gates | ../testcases/AQFP.alib|
| verilog | netlist of the circuit | ../testcases/neuron_yosysed_maj_balanced/neuron_yosysed_maj_balanced.v |
| row_adjust_rounds | number of rounds for iteratively updating placement stage | 5 |
| bfr_insert_rounds | number of rounds for buffer insertion of wirelength violation| 5 |
| split_threshold | thresholds of wirelength violaiton to split the row | {20, 15, 15, 10, 10, 5, 5, 3, 3, 2, 2, 1, 1} |
| il_unit | unit for .il files | 0.0125 |

### <cell_library>.alib


# Run Tests
Under build folder, run:
```bash
~$ ctest
```
