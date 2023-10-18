# *****************************************************************************
# Lab 6: Evaluation Metrics ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# The choice of evaluation metric depends on the specific problem,
# the characteristics of the data, and the goals of the modeling task.
# It's often a good practice to use multiple evaluation metrics to gain a more
# comprehensive understanding of a model's performance.

# There are several evaluation metrics that can be used to evaluate algorithms.
# The default metrics used are:
## (1) "Accuracy" for classification problems and
## (2) "RMSE" for regression problems

# Accuracy is the percentage of correctly classified instances out of all
# instances. Accuracy is more useful in binary classification problems than
# in multi-class classification problems.

# On the other hand, Cohen's Kappa is similar to Accuracy however, it is more
# useful on classification problems that do not have an equal distribution of
# instances amongst the classes in the dataset.

# For example, instead of Red are 50 instances and Blue are 50 instances,
# the distribution can be that Red are 70 instances and Blue are 30 instances.

# STEP 1. Install and Load the Required Packages ----
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# 1. Accuracy and Cohen's Kappa ----
## 1.a. Load the dataset ----
data(PimaIndiansDiabetes)

## 1.b. Determine the Baseline Accuracy ----
# Identify the number of instances that belong to each class (distribution or
# class breakdown).

# The result should show that 65% tested negative and 34% tested positive
# for diabetes.

# This means that an algorithm can achieve a 65% accuracy by
# predicting that all instances belong to the class "negative".

# This in turn implies that the baseline accuracy is 65%.

pima_indians_diabetes_freq <- PimaIndiansDiabetes$diabetes
cbind(frequency =
        table(pima_indians_diabetes_freq),
      percentage = prop.table(table(pima_indians_diabetes_freq)) * 100)

## 1.c. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.75,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)

# We then train a Generalized Linear Model to predict the value of Diabetes
# (whether the patient will test positive/negative for diabetes).

# `set.seed()` is a function that is used to specify a starting point for the
# random number generator to a specific value. This ensures that every time you
# run the same code, you will get the same "random" numbers.
set.seed(7)
diabetes_model_glm <-
  train(diabetes ~ ., data = pima_indians_diabetes_train, method = "glm",
        metric = "Accuracy", trControl = train_control)

## 1.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show an accuracy of approximately 77% (slightly above the baseline
# accuracy) and a Kappa of approximately 49%.
print(diabetes_model_glm)

### Option 2: Compute the metric yourself using the test dataset ----
# A confusion matrix is useful for multi-class classification problems.
# Please watch the following video first: https://youtu.be/Kdsp6soqA7o

# The Confusion Matrix is a type of matrix which is used to visualize the
# predicted values against the actual Values. The row headers in the
# confusion matrix represent predicted values and column headers are used to
# represent actual values.

predictions <- predict(diabetes_model_glm, pima_indians_diabetes_test[, 1:8])
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         pima_indians_diabetes_test[, 1:9]$diabetes)
print(confusion_matrix)

### Option 3: Display a graphical confusion matrix ----

# Visualizing Confusion Matrix
fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

# 2. RMSE, R Squared, and MAE ----

# RMSE stands for "Root Mean Squared Error" and it is defined as the average
# deviation of the predictions from the observations.

# R Squared (R^2) is also known as the "coefficient of determination".
# It provides a goodness of fit measure for the predictions to the
# observations.

# NOTE: R Squared (R^2) is a value between 0 and 1 such that
# 0 refers to "no fit" and 1 refers to a "perfect fit".

## 2.a. Load the dataset ----
data(longley)
summary(longley)
longley_no_na <- na.omit(longley)

## 2.b. Split the dataset ----
# Define a train:test data split of the dataset. Such that 10/16 are in the
# train set and the remaining 6/16 observations are in the test set.

# In this case, we split randomly without using a predictor variable in the
# caret::createDataPartition function.

# For reproducibility; by ensuring that you end up with the same
# "random" samples
set.seed(7)

# We apply simple random sampling using the base::sample function to get
# 10 samples
train_index <- sample(1:dim(longley)[1], 10) # nolint: seq_linter.
longley_train <- longley[train_index, ]
longley_test <- longley[-train_index, ]

## 2.c. Train the Model ----
# We apply bootstrapping with 1,000 repetitions
train_control <- trainControl(method = "boot", number = 1000)

# We then train a linear regression model to predict the value of Employed
# (the number of people that will be employed given the independent variables).
longley_model_lm <-
  train(Employed ~ ., data = longley_train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)

## 2.d. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show an RMSE value of approximately 4.3898 and
# an R Squared value of approximately 0.8594
# (the closer the R squared value is to 1, the better the model).
print(longley_model_lm)

### Option 2: Compute the metric yourself using the test dataset ----
predictions <- predict(longley_model_lm, longley_test[, 1:6])

# These are the 6 values for employment that the model has predicted:
print(predictions)

#### RMSE ----
rmse <- sqrt(mean((longley_test$Employed - predictions)^2))
print(paste("RMSE =", rmse))

#### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((longley_test$Employed - predictions)^2)
print(paste("SSR =", ssr))

#### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((longley_test$Employed - mean(longley_test$Employed))^2)
print(paste("SST =", sst))

#### R Squared ----
# We then use SSR and SST to compute the value of R squared
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", r_squared))

#### MAE ----
# MAE measures the average absolute differences between the predicted and
# actual values in a dataset. MAE is useful for assessing how close the model's
# predictions are to the actual values.

# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - longley_test$Employed)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))

# 3. Area Under ROC Curve ----
# Area Under Receiver Operating Characteristic Curve (AUROC) or simply
# "Area Under Curve (AUC)" or "ROC" represents a model's ability to
# discriminate between two classes.

# ROC is a value between 0.5 and 1 such that 0.5 refers to a model with a
# very poor prediction (essentially a random prediction; 50-50 accuracy)
# and an AUC of 1 refers to a model that predicts perfectly.

# ROC can be broken down into:
## Sensitivity ----
#         The number of instances from the first class (positive class)
#         that were actually predicted correctly. This is the true positive
#         rate, also known as the recall.
## Specificity ----
#         The number of instances from the second class (negative class)
#         that were actually predicted correctly. This is the true negative
#         rate.

## 3.a. Load the dataset ----
data(PimaIndiansDiabetes)
## 3.b. Determine the Baseline Accuracy ----
# The baseline accuracy is 65%.

pima_indians_diabetes_freq <- PimaIndiansDiabetes$diabetes
cbind(frequency =
        table(pima_indians_diabetes_freq),
      percentage = prop.table(table(pima_indians_diabetes_freq)) * 100)

## 3.c. Split the dataset ----
# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.8,
                                   list = FALSE)
pima_indians_diabetes_train <- PimaIndiansDiabetes[train_index, ]
pima_indians_diabetes_test <- PimaIndiansDiabetes[-train_index, ]

## 3.d. Train the Model ----
# We apply the 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

# We then train a k Nearest Neighbours Model to predict the value of Diabetes
# (whether the patient will test positive/negative for diabetes).

set.seed(7)
diabetes_model_knn <-
  train(diabetes ~ ., data = pima_indians_diabetes_train, method = "knn",
        metric = "ROC", trControl = train_control)

## 3.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show a ROC value of approximately 0.76 (the closer to 1,
# the higher the prediction accuracy) when the parameter k = 9
# (9 nearest neighbours).

print(diabetes_model_knn)

### Option 2: Compute the metric yourself using the test dataset ----
#### Sensitivity and Specificity ----
predictions <- predict(diabetes_model_knn, pima_indians_diabetes_test[, 1:8])
# These are the values for diabetes that the
# model has predicted:
print(predictions)
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         pima_indians_diabetes_test[, 1:9]$diabetes)

# We can see the sensitivity (≈ 0.86) and the specificity (≈ 0.60) below:
print(confusion_matrix)

#### AUC ----
# The type = "prob" argument specifies that you want to obtain class
# probabilities as the output of the prediction instead of class labels.
predictions <- predict(diabetes_model_knn, pima_indians_diabetes_test[, 1:8],
                       type = "prob")

# These are the class probability values for diabetes that the
# model has predicted:
print(predictions)

# "Controls" and "Cases": In a binary classification problem, you typically
# have two classes, often referred to as "controls" and "cases."
# These classes represent the different outcomes you are trying to predict.
# For example, in a medical context, "controls" might represent patients without
# a disease, and "cases" might represent patients with the disease.

# Setting the Direction: The phrase "Setting direction: controls < cases"
# specifies how you define which class is considered the positive class (cases)
# and which is considered the negative class (controls) when calculating
# sensitivity and specificity.
roc_curve <- roc(pima_indians_diabetes_test$diabetes, predictions$neg)

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)

# 4. Logarithmic Loss (LogLoss) ----
# Logarithmic Loss (LogLoss) is an evaluation metric commonly used for
# assessing the performance of classification models, especially when the model
# provides probability estimates for each class.

# LogLoss measures how well the predicted probabilities align with the true
# binary outcomes.

# In *binary classification*, the LogLoss formula for a single observation is:
# LogLoss = −(y log(p) + (1 − y)log(1 − p))

# Where:
# [*] y is the true binary label (0 or 1).
# [*] p is the predicted probability of the positive class.

# The LogLoss formula computes the logarithm of the predicted probability for
# the true class (if y = 1) or the logarithm of the predicted probability for
# the negative class (if y = 0), and then sums the results.

# A lower LogLoss indicates better model performance, where perfect predictions
# result in a LogLoss of 0.

########################### ----
## 4.a. Load the dataset ----
data(iris)

## 4.b. Train the Model ----
# We apply the 5-fold repeated cross validation resampling method
# with 3 repeats
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)
# This creates a CART model. One of the parameters used by a CART model is "cp".
# "cp" refers to the "complexity parameter". It is used to impose a penalty to
# the tree for having too many splits. The default value is 0.01.
iris_model_cart <- train(Species ~ ., data = iris, method = "rpart",
                         metric = "logLoss", trControl = train_control)

## 4.c. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show that a cp value of ≈ 0 resulted in the lowest
# LogLoss value. The lowest logLoss value is ≈ 0.46.
print(iris_model_cart)

# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint

# References ----

## Kuhn, M., Wing, J., Weston, S., Williams, A., Keefer, C., Engelhardt, A., Cooper, T., Mayer, Z., Kenkel, B., R Core Team, Benesty, M., Lescarbeau, R., Ziem, A., Scrucca, L., Tang, Y., Candan, C., & Hunt, T. (2023). caret: Classification and Regression Training (6.0-94) [Computer software]. https://cran.r-project.org/package=caret # nolint ----

## Leisch, F., & Dimitriadou, E. (2023). mlbench: Machine Learning Benchmark Problems (2.1-3.1) [Computer software]. https://cran.r-project.org/web/packages/mlbench/index.html # nolint ----

## National Institute of Diabetes and Digestive and Kidney Diseases. (1999). Pima Indians Diabetes Dataset [Dataset]. UCI Machine Learning Repository. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database # nolint ----

## Robin, X., Turck, N., Hainard, A., Tiberti, N., Lisacek, F., Sanchez, J.-C., Müller, M., Siegert, S., Doering, M., & Billings, Z. (2023). pROC: Display and Analyze ROC Curves (1.18.4) [Computer software]. https://cran.r-project.org/web/packages/pROC/index.html # nolint ----

## Wickham, H., François, R., Henry, L., Müller, K., Vaughan, D., Software, P., & PBC. (2023). dplyr: A Grammar of Data Manipulation (1.1.3) [Computer software]. https://cran.r-project.org/package=dplyr # nolint ----

## Wickham, H., Chang, W., Henry, L., Pedersen, T. L., Takahashi, K., Wilke, C., Woo, K., Yutani, H., Dunnington, D., Posit, & PBC. (2023). ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics (3.4.3) [Computer software]. https://cran.r-project.org/package=ggplot2 # nolint ----

# **Required Lab Work Submission** ----
## Part A ----
# Create a new file called
# "Lab6-Submission-EvaluationMetrics.R".
# Provide all the code you have used to demonstrate the classification and
# regression evaluation metrics we have gone through in this lab.
# This should be done on any datasets of your choice except the ones used in
# this lab.

## Part B ----
# Upload *the link* to your
# "Lab6-Submission-EvaluationMetrics.R" hosted
# on Github (do not upload the .R file itself) through the submission link
# provided on eLearning.

## Part C ----
# Create a markdown file called "Lab-Submission-Markdown.Rmd"
# and place it inside the folder called "markdown". Use R Studio to ensure the
# .Rmd file is based on the "GitHub Document (Markdown)" template when it is
# being created.

# Refer to the following file in Lab 1 for an example of a .Rmd file based on
# the "GitHub Document (Markdown)" template:
#     https://github.com/course-files/BBT4206-R-Lab1of15-LoadingDatasets/blob/main/markdown/BIProject-Template.Rmd # nolint

# Include Line 1 to 14 of BIProject-Template.Rmd in your .Rmd file to make it
# displayable on GitHub when rendered into its .md version

# It should have code chunks that explain all the steps performed on the
# datasets.

## Part D ----
# Render the .Rmd (R markdown) file into its .md (markdown) version by using
# knitR in RStudio.

# You need to download and install "pandoc" to render the R markdown.
# Pandoc is a file converter that can be used to convert the following files:
#   https://pandoc.org/diagram.svgz?v=20230831075849

# Documentation:
#   https://pandoc.org/installing.html and
#   https://github.com/REditorSupport/vscode-R/wiki/R-Markdown

# By default, Rmd files are open as Markdown documents. To enable R Markdown
# features, you need to associate *.Rmd files with rmd language.
# Add an entry Item "*.Rmd" and Value "rmd" in the VS Code settings,
# "File Association" option.

# Documentation of knitR: https://www.rdocumentation.org/packages/knitr/

# Upload *the link* to "Lab-Submission-Markdown.md" (not .Rmd)
# markdown file hosted on Github (do not upload the .Rmd or .md markdown files)
# through the submission link provided on eLearning.