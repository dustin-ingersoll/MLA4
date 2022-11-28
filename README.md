# MLA4
ML-7641 Assignment 4

Author: Dustin Ingersoll (dingersoll3)

Env: PyCharm 2021.3, python 3.8

Date: 11/27/2022


=== SUMMARY ===

Code link: https://github.com/dustin-ingersoll/MLA4

Included are the source code and graphs used in the analysis.


MLA4
|
|- graphs     ( the resulting graphs )
|
|- graphing.py     ( code for graphing run_stats )
|
|- policy_iteration.py     ( test code for PI)
|
|- q_learning.py     ( test code for Q-Learning )
|
|- requirements.txt      ( required imports to run project in python environment )
|
|- run.py      ( functions to run all tests consecutively )
|
|- value_iteration.py      ( test code for VI )



=== USABILITY ===


1) Install the requirements.txt into your python environment
2) install mdptoolbox with "pip install git+https://github.com/hiive/hiivemdptoolbox.git"
 *if scipy throws an error, run "pip uninstall scipy && pip install scipy"
3) Run the file "run.py". This will run all tests and replace the graphs currently in the project.
 * Stats are printed to the terminal

=== EDITING ===


Each file named after the MDP (policy_iteration.py, value_iteration.py, etc. ) will contain the code used to run each MDP task.

