Neural network mammography database analysis
============================================

## Needed software and packages:

* Python 3.6.x (preferably anaconda distribution, https://www.anaconda.com/download/)
* scikit-learn (already installed in anaconda distribution)
* jupyter and derived (already installed in anaconda distribution)
* imbalanced-learn (conda install -c glemaitre imbalanced-learn)

Script modifications should be made in the nnmammo.py, the python comments \# \<markdowncell\>, \# \<codecell\>, \# \<rawcell\> can be used to format the script and automatically generate a jupyter notebook. This configuration allows easier versioning and debugging.
* notebook generation (script -> notebook): python script2notebook.py

## Runing the project
After installing all dependencies, the project can be run  in two ways, from the nnmammmo.py script (simply run the script), or from the nnmammo.ipynb, that can be generated using the script2notebook.py, run the script2notebook.py and it will generate an updated notebook from the nnmammo.py script.
The notebook contains the script formated in cell with moduled conde that can be executed in parts.
To run the notebook execute the command jupyter notebook nnmammo.ipynb.
