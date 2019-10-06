
Objective

Predictive models built to predict the median price based on features such as Crime rate and Number of rooms etc. and identified the highest performing model based on specific evaluation metrics: Adjusted R-squared (Adj-R2) and Mean Squared Error (MSE).

Summary

The Boston dataset was used to predict the median price based on features such as Crime rate and Number of rooms etc. and identified the highest performing model based on specific evaluation metrics: Adjusted R-squared (Adj-R2) and Mean Squared Error (MSE). The Adj-R2 measures the goodness of fit of the model adjusted for the number of predictors while the MSE is an error measure where the squared error for each data point is calculated and summed across all the datapoints and a mean value is calculated. Error is the difference between the actual and predicted value. 

The standard Machine Learning Model building pipeline (as noted below) was used. The six distinct models (however over eight in total given sub-setting of features) varied in terms of complexity with the simplest model being the Linear Regression Model, then Penalized Linear Regression, Polynomial Linear Regression, Polynomial Penalized Linear Regression, Decision Trees, Random Forest. Finally, the model complexity peaked at an ensemble model called Gradient Boosted Machine. Note that a subset of features were used for some models and or the entire set of features.
 
Top Performer
– Gradient Boosted Machine (MSE: 11.144 and Adj-R2: 0.867) where all features were used.

2nd Top 
– Polynomial Penalized (Lasso) Linear Regression (MSE: 12.818 & Adj-R2: 0.83). 

Note that if interpretability was a high priority then I recommended choosing the 2nd Top performer (Lasso Regression -Polynomial using all features) since Gradient Boosted Machines (similar to other ensemble models) are much harder to explain the basis for the predictions. 


The six distinct machine learning models which applied regression techniques and were evaluated were:
1. Linear Regression 
   - using subset of features

2. Polynomial Linear Regression
   - using subset of features

3. Penalized Linear Regression - Lasso Regression
   - using subset of features
   - using all features
   - using polynomial transformation on all features

4. Decision Regression Tree

Ensemble Models :
5. Random Forest
6. Gradient Boosted Tree

(i)	Linear Regression
A statistical model which measures the linear relationship of the continuous dependent variable and multiple predictor (independent) variables. Note the linear relationship is defined with respect to the model parameters (feature coefficients). The prediction is a linear additive of the parameters of the predictive variables. 

(ii)	Polynomial Linear Regression
A variant of linear regression where the features have been transformed to polynomial features of n degrees (eg. X2 where n=2 or X3 where n=3 etc). This is effected in order to fit a non-linear function in cases where the underlying pattern of the data appears to be non-linear. It is still considered Linear Regression since the fitted line/function is still linear in its parameters. 

(iii)	Penalized (Lasso) Linear Regression 
Another variant of a linear regression model where the model parameters are shrunk towards zero. In this case of Lasso regularization, the parameters can actually be shrunken all the way to zero (for Ridge Regression, the shrinkage is towards zero but never zero). In light of this, this model innately conducts feature sub-setting. 

(iv)	Decision Tree 
A hierarchical tree-based statistical method that uses a top-down greedy search through the input space and iteratively partitions the input space based on the feature which is the best at reducing the mean squared error at that level in the tree (before versus after a split of the current data on that feature) in classification the best predictor is assessed based on an impurity measure (eg. Entropy). 

(v)	Random Forest
This is an ensemble method that uses a decision tree as the base learner. The method randomizes by row and column thereby randomly selecting a subset of the dataset to grow a tree (a learner) and randomly selects the candidate features from which a best feature will be chosen. It uses bootstrap aggregation (Bagging – random sampling with replacement) to derive the final prediction using multiple learners. Bagging reduces the variance of the model given that we average over the individual learners. Note in Random forest we generate the base learners in parallel, one independent of the other.  This is in contrast to gradient boosting which we will delve into in the next section below.

(vi)	Gradient Boosted Trees 
An ensemble method where more than one decision models work together to produce best results with each model that is gradually added (sequentially in contrast to random forest were multiple leaners are generated in parallel), trying to correct the mistakes of the previous model.  


