# cs534-final-project : Applying Machine Learning to Understand the Relationship Between Education and Crime

## Overview and Background:

Previous works have shown a clear relationship between various metrics of education spending and crime rates (Lochner et al. 2004, Gonzalez 2015, McClendon et al. 2015). But these approaches were all focused on macro-scale spending and crime data, rather than the direct effects of _local_ education spend on _local_ crime. With the U.S. correctional facilities overflowing and currently allocating more federal funds to crime prevention than education, gaining a better understanding of these relationships is an urgent and important pursuit. To try to better understand how local school districts can most effectively allocate their budgets for greatest crime reduction, we explore local spending effects in this open source code base, taking advantage of two rich datasets:
- **Georgia Bureau of Investigation - Georgia  Uniform Crime Reporting Program**
  - Contains crime data for all U.S. counties by year
  - Covers a range of crimes (violent crimes, theft, etc.) across 2009-2017
- **U.S. Census Bureau - Annual Survey of School System Finances**
  - Contains spend data for all U.S. school districts by year 
  - Covers a range of spending figures from 1992-2008

We invite you to explore the scripts and functions we have written, so you can get a sense of these relationships in the state of Georgia! We provide several pre-written functions in this repo that allow you to isolate the most relevant features and reveal the important relationship between _education spending_ and _future crime_. The machine learning approaches we have implemented in this repo are: Principal Component Analysis (PCA), Non-negative Matrix Factorization (NMF), Support Vector Machine (SVM), Gradient Boosting (XGBoost), Lasso regression, and Least Squares. 

We provide scripts for plotting and comparison of modeling performance and feature selection across these approaches.

## How to Use:
Below is a list of the main functions and scripts in our repo, and a brief description of what each one does.
  1. **main_data_exploration.py**
      - Function may be run from top to bottom, as is, to get R2 values for the best Lasso, PCA, and NMF parameters.
      - You may also plot the results across many fits by enabling the _plot_time_delay_results()_ function at the end of the script, if desired
  2. **data_exploration_all_plots.py**
      - This function is very similar to the above, but includes plotting by default 
      - This will output all grid search results and R2 values across numbers of dropped features that result in the "Best Fit" solutions
  3. **human_selected_variables.py**
      - This function loads all the data and runs a similar analysis on variables selected by us to try to "guess" which variables would be most predictive.
      - Feel free to try your own guesses!
  4. **edu_data_load.py**
      - This function will load the education spending data into a Pandas DataFrame for use in other scripts
  5. **data_load.py**
      - This function loads both the crime data and the education spending data into a common output DataFrame.
      - It also contains a function for scaling all spending features by the total revenue for each school district (for normalization)


## Evaluation and Discussion:

Out of 131 education spend categories, the final results yield 10-15 features that significantly affect violent crime rates (murder, rape, aggravated assault). A few of these features include tuition fees to pupils/parents, school lunch revenues, total employee benefits (instruction), and payments to other school systems. LASSO regression was used as a baseline dimensional reduction with SVM and NMF following up in series. The baseline time delay was 11 years as education spend was averaged from all K-12 years and a subsequent addition of 0 to 8 years was evaluated across all reduction techiniques using SSE as the error measurement. 15 years total proved to show the highest correlation between education spend and violet crime data. 

![](cs534-final-project/blob/main/Presentation/reduction.png?raw=true)

A handpicked set of 30 features also ran through these techniques. Features such as Instructor salaries, employee benefits, property taxes, and federal revenue through state seemed intuitively important to potential violent crime potential and were measured against the total lasso with no handpicking with vastly different results. Some features did not total 1% of spend and other features, like employee benefits actually turned out to have the opposite relationship as intuitively thought (more benefits actually --> more violent crime). 

Of features that totaled more than 1% of federal spend, 3 stood out. Total salaries and wages - student transportation, total salaries and wages - instructional staff support, and total salaries and wages - food services. Food services had a positive beta of 35, suggesting that with each 1% increase compared to Total Revenue, a random county should see an *increase* of 35 violent crimes.  
