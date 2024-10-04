# OCTDL Training

Use Python version: 3.9

Install the requirements via the install.py script for the pytorch version used to generate the results.

You need to download the OCTDL dataset from here: https://data.mendeley.com/datasets/sncdhf53xc/1

Put the files in the root folder with the following structure:

```
/OCTDL
|--OCTDL_labels.csv
|--/AMD
|--/DME
   ...
```

The experiments are contained in the run_\*.py files.
The respective script to evaluate them are the eval_\*.py files.