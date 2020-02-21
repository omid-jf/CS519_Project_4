python main.py

To improve the performance of the regressors, we suggest to use feature selection first and choose the attributes which
have the strongest relation with the class variable. This is called Feature Selection, and SKlearn has a library for it
called LassoCV. We will use this library and run the regressors only for the California Renewable Production dataset.