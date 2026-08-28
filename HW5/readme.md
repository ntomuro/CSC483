# :deciduous_tree: CSC 483 HW#5 Backprop Hyperparameters

## :rocket: Note

Assignment page: https://condor.depaul.edu/ntomuro/courses/483/2026fall/assign/HW5/hw5-2026fall.html

## :herb: Setup: Your Development Environment

Before you write any code, set up your environment exactly as specified below.
This matters: because the assignment asks you to compare your numeric output to
the sample output given, a mismatched library version can shift results in the
4th decimal place and make correct code look wrong.

### Python and package versions

Use a **Python 3.11–3.13** installation. Run `python --version`
to confirm. Do not install the newest Python release available on the day
you start — numpy/pandas/matplotlib wheels for a just-released Python
version sometimes lag by a few months, and installation will fail or fall back
to building from source.

Install the exact package set with the provided [requirements.txt](requirements.txt):

```
pip install -r requirements.txt
```

(If you use conda, create an environment first: `conda create -n csc483 python=3.12`,
`conda activate csc483`, then run the pip command above inside it.) Note the
**`import-ipynb`** package in the file — that is what lets a notebook
`import` another notebook (`NN483_network2.ipynb`) as if it were a plain module.

### Folder layout

Put `NN483_network2.ipynb`, all the data files (`iris.csv`,
`iris-423.dat`, `iris4-20-7-3.dat`, `iris-train-2.csv`,
`iris-test-2.csv`), and every notebook you write in
*one flat folder*.

*Do not hardcode an absolute path* anywhere in your
notebook (e.g. no `os.chdir('/content/drive/My Drive/...')` or
`C:\...` paths). Your submitted notebooks must run top-to-bottom
with zero edits when placed in a folder next to the data files — that is
literally how it will be graded. If you need the current directory for any
reason, use a relative reference (`'.'`), never a path specific to
your machine or Drive account.

### Note: Editing NN483_network2.ipynb without restarting the kernel

Because `import_ipynb` only executes the network notebook the
**first** time it is imported, an application notebook that already ran
`import NN483_network2 as network2` will **not** pick up
later edits you make to `NN483_network2.ipynb`. After you change the network
notebook, **restart the kernel** of whichever application notebook is using it
and re-run the cells from the top (including the `import` cell) —
otherwise you will be testing stale code and get confused by results that don't
match your latest edits.

### (Optional) Smoke test

Before writing any of your own code, run this cell (after installing
`requirements.txt`, with `iris.csv` in the same folder) and confirm
your output matches exactly. If it doesn't, your environment is misconfigured and you
should fix that first — do not proceed until this matches.

```python
import sys, numpy, pandas
print("Python: ", sys.version.split()[0])
print("numpy:  ", numpy.__version__)
print("pandas: ", pandas.__version__)

!pip install import-ipynb
import import_ipynb
import NN483_network2 as network2

data = network2.my_load_csv('iris.csv', 4, 3)
print("Loaded", len(data), "instances")
print("x0 shape:", data[0][0].shape, " y0 shape:", data[0][1].shape)
print("x0:", data[0][0].ravel())
print("y0:", data[0][1].ravel())
```

Expected output (library version numbers may differ slightly — that's
fine as long as they satisfy the minimums in `requirements.txt`;
the loaded data must match exactly, since it depends on the fixed shuffle seed):

```
Loaded 150 instances
x0 shape: (4, 1)  y0 shape: (3, 1)
x0: [5.4 3.9 1.3 0.4]
y0: [1 0 0]
```

### Using Google Colab instead

If you don't want to install anything locally, Colab still works. Either upload
`NN483_network2.ipynb` and the data files with Colab's file-upload panel
(left sidebar → folder icon → upload), or clone your files from the
course repo with `!git clone https://github.com/ntomuro/CSC483`, then
`!pip install import-ipynb` and `import NN483_network2 as network2`
exactly as above. numpy/pandas are already installed on Colab. You no longer need
to mount Google Drive or hardcode a Drive path — uploaded/cloned files land in
Colab's local working directory, so a plain relative filename (e.g. `'iris.csv'`)
is all you need.
