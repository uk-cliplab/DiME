Variational_bounds is a notebook intended to be ran in Jupyter Lab. 
There is a little bit of simple setup required

Make a new conda env accessible by jupyter lab

STEP 1: Make a new conda env:   conda create --name <name>

STEP 2: Activate the conda env:  conda activate <name>

STEP 3: Revert to python 3.6:    conda install python=3.6

STEP 4: conda install ipykernel

STEP 5: ipython kernel install --name=<name> --user

STEP 6: pip install -r requirements.txt

Now this new conda kernel will appear in Jupyterlab. If it doesn't try refreshing the page. Run the notebook with the kernel
