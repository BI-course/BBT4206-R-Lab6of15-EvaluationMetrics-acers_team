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
library(readr)
students_performance_dataset <- read_csv("data/students_performance_dataset.csv")
View(students_performance_dataset)
data(students_performance_dataset) #let's use cols 7:17 -> determine whether student pass or fail

## 1.b. Determine the Baseline Accuracy ---
students_performance_dataset_freq <- students_performance_dataset$`TOTAL = Coursework TOTAL + EXAM (100%)`
cbind(frequency =
        table(students_performance_dataset_freq),
      percentage = prop.table(table(students_performance_dataset_freq)) * 100)

## 1.c. Split the dataset ----
train_index <- createDataPartition(students_performance_dataset$`TOTAL = Coursework TOTAL + EXAM (100%)`,
                                   p = 0.75,
                                   list = FALSE)
students_performance_dataset_train <- students_performance_dataset[train_index, ]
students_performance_dataset_test <- students_performance_dataset[-train_index, ]

## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)

# We then train a Generalized Linear Model to predict the value of TOTAL MARKS 
# (whether the student will pass or fail the text based on totalmarks
# `set.seed()` is a function that is used to specify a starting point for the
# random number generator to a specific value. This ensures that every time you
# run the same code, you will get the same "random" numbers.
set.seed(7) #you can use any values the point is to introduce the randomness 
TOTAL_model_glm <-
  train(TOTAL ~ ., data = students_performance_dataset_train, method = "glm",
        metric = "Accuracy", trControl = train_control) #all other variables ~ .).
