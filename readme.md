# Lecture Notes on Real-Time Geological Inversion for Decision-Making and Stratigraphic Inversion Exercise

## Lecture notes

Lecture notes are in [the content folder](/content).

## Overview of files

### Utility scripts

#### `utils.py`
Set of useful functions mapping expected stratigraphy to data.

#### `my_curve_data.py`
Provides an offset log

#### `my_trajectory_data.py`
Provides a boundary "B"-function

### Some starting points

#### Testing that your cuatom function works
`sequential_runner.py` lets you import your custom function and show the resulting solution.

#### `my_curve_fit.py`
Contains a sequential fitting using `curvefit`

#### `my_curve_fit_with_minimize.py`
Contains a sequential fitting with Tikhonov Regularization using `minimize`


### Competitoin `competition_plotter.py`
Is the file that would be used to compare the results

## How to add your solution?
1. Clone this repository
2. Create your solution that fits the curve sequentially to data in `submissions`
3. Add your solution to `competition_plotter.py` following the instructions
- `todo 1. import your solution from folder submissions with a unique id`
- `todo 2. add your solver to dictionary`
4. Create a pull request



