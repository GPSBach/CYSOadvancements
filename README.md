# CYSOadvancements
Chicago Youth Symphony Orchestra (CYSO) college advancement statistics for diversity initiative

The CYSO wanted to examine the demographic data of their graduating seniors, who have reported their chosen university and major upon departing the organization.  The CYSO was able to provide information such as gender, ethnicity, and level of financial aid for students that graduated between 2012 and 2018, and wants to know which factors are most important in determining which students will go on to major in music in college (the CYSO has a 100% college attendance rate for their graduating students).  This data was anonymized before processing, to remove any personally identifying information about CYSO students.

A detailed summary of this project is available [on my blog](https://extraordinaryleastsquares.com/2018/11/02/increasing-diversity-in-the-professional-orchestra-pipeline/?frame-nonce=9cb38c1e44#)

# Structure
./src - python executables
> parse_data.py reads in anonymized data and returns .csv files formatted for regression modeling
> fit_data_iterate.py fits the data using a suite of models with a Monte Carlo cross validation scheme
> plot_data.py has functions to parse and plot model results

./workbooks - jupyter notebooks for working on code

./data - .csv files containing project data and model pickles

./plots - output directory for plot_data.py
