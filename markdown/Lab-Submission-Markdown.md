---
editor_options: 
  markdown: 
    wrap: 72
  chunk_output_type: console
output: 
  html_document: 
    keep_md: yes
---

------------------------------------------------------------------------

title: "Business Intelligence Lab Submission Markdown" author: "Acers
Team" date: "October 19, 2023"

output: github_document: toc: yes toc_depth: 4 fig_width: 6 fig_height:
4 df_print: default editor_options: chunk_output_type: console ---

# Student Details

+----------------------------------+----------------------------------+
| **Student ID Numbers and Names   | *\<list one student name, group, |
| of Group Members**               | and ID per line; you should be   |
|                                  | between 2 and 5 members per      |
|                                  | group\>*                         |
|                                  |                                  |
|                                  | 1.  122790 - C - Bwalley         |
|                                  |     Nicholas                     |
|                                  |                                  |
|                                  | 2.  133834 - C - Mongare Sarah   |
|                                  |                                  |
|                                  | 3.  133928 - C - Cheptoi         |
|                                  |     Millicent                    |
|                                  |                                  |
|                                  | 4.  134879 - C - Tulienge Lesley |
|                                  |                                  |
|                                  | 5.  124461 - C - Kinya Angela    |
+----------------------------------+----------------------------------+
| **GitHub Classroom Group Name**  | Acers Team                       |
+----------------------------------+----------------------------------+
| **Course Code**                  | BBT4206                          |
+----------------------------------+----------------------------------+
| **Course Name**                  | Business Intelligence II         |
+----------------------------------+----------------------------------+
| **Program**                      | Bachelor of Business Information |
|                                  | Technology                       |
+----------------------------------+----------------------------------+
| **Semester Duration**            | 21^st^ August 2023 to 28^th^     |
|                                  | November 2023                    |
+----------------------------------+----------------------------------+

# Understanding the Dataset (Exploratory Data Analysis (EDA))

## Loading the Dataset

### Source:

The dataset that was used can be downloaded here:
<https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset>\*

### Reference:

*\<Cite the dataset here using APA\>\
Refer to the APA 7th edition manual for rules on how to cite datasets:
<https://apastyle.apa.org/style-grammar-guidelines/references/examples/data-set-references>*

# STEP 1. Install and Load the Required Packages ----

``` r
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

```         
## Loading required package: ggplot2
```

``` r
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

```         
## Loading required package: caret
```

```         
## Loading required package: lattice
```

``` r
## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

```         
## Loading required package: mlbench
```

``` r
## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

```         
## Loading required package: pROC
```

```         
## Type 'citation("pROC")' for a citation.
```

```         
## 
## Attaching package: 'pROC'
```

```         
## The following objects are masked from 'package:stats':
## 
##     cov, smooth, var
```

``` r
## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

```         
## Loading required package: dplyr
```

```         
## 
## Attaching package: 'dplyr'
```

```         
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```         
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

# 1. Accuracy and Cohen's Kappa ----

## 1.a. Load the dataset ----

``` r
library(readr)
heart_attack_dataset <- read_csv("C:/Users/NICK BWALLEY/OneDrive - Strathmore University/MyStrath/BBIT/4.2/Business Intelligence II - Dr. Allan Omondi/BI2-Labs/BBT4206-R-Lab6of15-EvaluationMetrics-acers_team/data/heart_attack_dataset.csv")
```

```         
## Rows: 1319 Columns: 9
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr (1): class
## dbl (8): age, gender, impluse, pressurehight, pressurelow, glucose, kcm, tro...
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
View(heart_attack_dataset)
# data(heart_attack_dataset) 
# sapply(heart_attack_dataset, class) #datatypes of variables


## 1.b. Determine the Baseline Accuracy ----
heart_attack_dataset_freq <- heart_attack_dataset$class
cbind(frequency =
        table(heart_attack_dataset_freq),
      percentage = prop.table(table(heart_attack_dataset_freq)) * 100)
```

```         
##          frequency percentage
## negative       509   38.58984
## positive       810   61.41016
```

``` r
## 1.c. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(heart_attack_dataset$class,
                                   p = 0.75,
                                   list = FALSE)
heart_attack_dataset_train <- heart_attack_dataset[train_index, ]
heart_attack_dataset_test <- heart_attack_dataset[-train_index, ]


## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)


# `set.seed()` is a function that is used to specify a starting point for the
# random number generator to a specific value. This ensures that every time you
# run the same code, you will get the same "random" numbers.
set.seed(7) #you can use any values the point is to introduce the randomness 
class_model_glm <-
  train(class ~ ., data = heart_attack_dataset_train, method = "glm",
        metric = "Accuracy", trControl = train_control) #all other variables ~ .
```

```         
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

``` r
## 1.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
print(class_model_glm)
```

```         
## Generalized Linear Model 
## 
## 990 samples
##   8 predictor
##   2 classes: 'negative', 'positive' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 792, 791, 792, 793, 792 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.8252609  0.6360275
```

# 2. RMSE, R Squared, and MAE ----

``` r
# PLEASE NOTE: 
## THIS STEP IS NOT APPLICABLE IN OUR DATASET SO IN THIS SECTION WE ARE NOT GOING TO PERFORM THE COMPUTATIONS IN THIS SECTION. 
```

# 3. Area Under ROC Curve ----

Area Under Receiver Operating Characteristic Curve (AUROC) or simply
"Area Under Curve (AUC)" or "ROC" represents a model's ability to
discriminate between two classes.

``` r
## 3.b. Determine the Baseline Accuracy ----
heart_attack_dataset_freq <- heart_attack_dataset$class
cbind(frequency =
        table(heart_attack_dataset_freq),
      percentage = prop.table(table(heart_attack_dataset_freq)) * 100)
```

```         
##          frequency percentage
## negative       509   38.58984
## positive       810   61.41016
```

``` r
## 3.c. Split the dataset ----
# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(heart_attack_dataset$class,
                                   p = 0.8,
                                   list = FALSE)
heart_attack_dataset_train <- heart_attack_dataset[train_index, ]
heart_attack_dataset_test <- heart_attack_dataset[-train_index, ]

## 3.d. Train the Model ----
# We apply the 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)


set.seed(7)
class_model_knn <-
  train(class ~ ., data = heart_attack_dataset_train, method = "knn",
        metric = "ROC", trControl = train_control)

## 3.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----

print(class_model_knn)
```

```         
## k-Nearest Neighbors 
## 
## 1056 samples
##    8 predictor
##    2 classes: 'negative', 'positive' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 950, 951, 950, 951, 950, 950, ... 
## Resampling results across tuning parameters:
## 
##   k  ROC        Sens       Spec     
##   5  0.6389927  0.4876829  0.6788462
##   7  0.6527297  0.4510976  0.7267067
##   9  0.6696653  0.4682317  0.7467308
## 
## ROC was used to select the optimal model using the largest value.
## The final value used for the model was k = 9.
```

``` r
#### AUC ----
# The type = "prob" argument specifies that you want to obtain class
# probabilities as the output of the prediction instead of class labels.
predictions <- predict(class_model_knn, heart_attack_dataset_test[, 1:8],
                       type = "prob")


print(predictions)
```

```         
##      negative  positive
## 1   0.1111111 0.8888889
## 2   0.0000000 1.0000000
## 3   0.2222222 0.7777778
## 4   0.3333333 0.6666667
## 5   0.3333333 0.6666667
## 6   0.5555556 0.4444444
## 7   0.4444444 0.5555556
## 8   0.3333333 0.6666667
## 9   0.3333333 0.6666667
## 10  0.4444444 0.5555556
## 11  0.3333333 0.6666667
## 12  0.1111111 0.8888889
## 13  0.0000000 1.0000000
## 14  0.4444444 0.5555556
## 15  0.5555556 0.4444444
## 16  0.3333333 0.6666667
## 17  0.2222222 0.7777778
## 18  0.4444444 0.5555556
## 19  0.0000000 1.0000000
## 20  0.2222222 0.7777778
## 21  0.6666667 0.3333333
## 22  0.0000000 1.0000000
## 23  0.5555556 0.4444444
## 24  0.4444444 0.5555556
## 25  0.3333333 0.6666667
## 26  0.6666667 0.3333333
## 27  0.6666667 0.3333333
## 28  0.5555556 0.4444444
## 29  0.3333333 0.6666667
## 30  0.5555556 0.4444444
## 31  0.5555556 0.4444444
## 32  0.4444444 0.5555556
## 33  0.3333333 0.6666667
## 34  0.6666667 0.3333333
## 35  0.1111111 0.8888889
## 36  0.1111111 0.8888889
## 37  0.1111111 0.8888889
## 38  0.5555556 0.4444444
## 39  0.4444444 0.5555556
## 40  0.0000000 1.0000000
## 41  0.2222222 0.7777778
## 42  0.0000000 1.0000000
## 43  0.6666667 0.3333333
## 44  0.3333333 0.6666667
## 45  0.6666667 0.3333333
## 46  0.6666667 0.3333333
## 47  0.3333333 0.6666667
## 48  0.2222222 0.7777778
## 49  1.0000000 0.0000000
## 50  0.5555556 0.4444444
## 51  0.3333333 0.6666667
## 52  0.4444444 0.5555556
## 53  0.2222222 0.7777778
## 54  0.4444444 0.5555556
## 55  0.1111111 0.8888889
## 56  0.5555556 0.4444444
## 57  0.7777778 0.2222222
## 58  0.2222222 0.7777778
## 59  0.3333333 0.6666667
## 60  0.0000000 1.0000000
## 61  0.4444444 0.5555556
## 62  0.3333333 0.6666667
## 63  0.7777778 0.2222222
## 64  0.5555556 0.4444444
## 65  0.4444444 0.5555556
## 66  0.4444444 0.5555556
## 67  0.4444444 0.5555556
## 68  0.5555556 0.4444444
## 69  0.1111111 0.8888889
## 70  0.5555556 0.4444444
## 71  0.4444444 0.5555556
## 72  0.3333333 0.6666667
## 73  0.2222222 0.7777778
## 74  0.3333333 0.6666667
## 75  0.5555556 0.4444444
## 76  0.2222222 0.7777778
## 77  0.5555556 0.4444444
## 78  0.8888889 0.1111111
## 79  0.4444444 0.5555556
## 80  0.1111111 0.8888889
## 81  0.3333333 0.6666667
## 82  0.3333333 0.6666667
## 83  0.7777778 0.2222222
## 84  0.6666667 0.3333333
## 85  0.0000000 1.0000000
## 86  0.4444444 0.5555556
## 87  0.0000000 1.0000000
## 88  0.6666667 0.3333333
## 89  0.4444444 0.5555556
## 90  0.8888889 0.1111111
## 91  0.4444444 0.5555556
## 92  0.5555556 0.4444444
## 93  0.5555556 0.4444444
## 94  0.1111111 0.8888889
## 95  0.1111111 0.8888889
## 96  0.2222222 0.7777778
## 97  0.3333333 0.6666667
## 98  0.2222222 0.7777778
## 99  0.2222222 0.7777778
## 100 0.3333333 0.6666667
## 101 0.6666667 0.3333333
## 102 0.4444444 0.5555556
## 103 0.6666667 0.3333333
## 104 0.6666667 0.3333333
## 105 0.6666667 0.3333333
## 106 0.3333333 0.6666667
## 107 0.3333333 0.6666667
## 108 0.4444444 0.5555556
## 109 0.4444444 0.5555556
## 110 0.4444444 0.5555556
## 111 0.4444444 0.5555556
## 112 0.4444444 0.5555556
## 113 0.2222222 0.7777778
## 114 0.2222222 0.7777778
## 115 0.4444444 0.5555556
## 116 0.4444444 0.5555556
## 117 0.8888889 0.1111111
## 118 0.0000000 1.0000000
## 119 0.3333333 0.6666667
## 120 0.5555556 0.4444444
## 121 0.0000000 1.0000000
## 122 0.4444444 0.5555556
## 123 0.5555556 0.4444444
## 124 0.3333333 0.6666667
## 125 0.4444444 0.5555556
## 126 0.0000000 1.0000000
## 127 0.4444444 0.5555556
## 128 0.3333333 0.6666667
## 129 0.6666667 0.3333333
## 130 0.4444444 0.5555556
## 131 0.5555556 0.4444444
## 132 0.4444444 0.5555556
## 133 0.4444444 0.5555556
## 134 0.0000000 1.0000000
## 135 0.4444444 0.5555556
## 136 0.3333333 0.6666667
## 137 0.6666667 0.3333333
## 138 0.5555556 0.4444444
## 139 0.1111111 0.8888889
## 140 0.5555556 0.4444444
## 141 0.1111111 0.8888889
## 142 0.2222222 0.7777778
## 143 0.4444444 0.5555556
## 144 0.7777778 0.2222222
## 145 0.6666667 0.3333333
## 146 0.6666667 0.3333333
## 147 0.0000000 1.0000000
## 148 0.1111111 0.8888889
## 149 0.4444444 0.5555556
## 150 0.4444444 0.5555556
## 151 0.4444444 0.5555556
## 152 0.4444444 0.5555556
## 153 0.4444444 0.5555556
## 154 0.2222222 0.7777778
## 155 0.6666667 0.3333333
## 156 0.3333333 0.6666667
## 157 0.6666667 0.3333333
## 158 0.6666667 0.3333333
## 159 0.2222222 0.7777778
## 160 0.3333333 0.6666667
## 161 0.2222222 0.7777778
## 162 0.3333333 0.6666667
## 163 0.4444444 0.5555556
## 164 0.2222222 0.7777778
## 165 0.7777778 0.2222222
## 166 0.5555556 0.4444444
## 167 0.1111111 0.8888889
## 168 0.3333333 0.6666667
## 169 0.4444444 0.5555556
## 170 0.1111111 0.8888889
## 171 0.2222222 0.7777778
## 172 0.4444444 0.5555556
## 173 0.7777778 0.2222222
## 174 0.4444444 0.5555556
## 175 0.4444444 0.5555556
## 176 0.5555556 0.4444444
## 177 0.7777778 0.2222222
## 178 0.0000000 1.0000000
## 179 0.6666667 0.3333333
## 180 0.3333333 0.6666667
## 181 0.3333333 0.6666667
## 182 0.4444444 0.5555556
## 183 0.4444444 0.5555556
## 184 0.5555556 0.4444444
## 185 0.3333333 0.6666667
## 186 0.5555556 0.4444444
## 187 0.3333333 0.6666667
## 188 0.0000000 1.0000000
## 189 0.5555556 0.4444444
## 190 0.4444444 0.5555556
## 191 0.0000000 1.0000000
## 192 0.2222222 0.7777778
## 193 0.2222222 0.7777778
## 194 0.7777778 0.2222222
## 195 0.4444444 0.5555556
## 196 0.2222222 0.7777778
## 197 0.5555556 0.4444444
## 198 0.1111111 0.8888889
## 199 0.4444444 0.5555556
## 200 0.3333333 0.6666667
## 201 0.3333333 0.6666667
## 202 0.2222222 0.7777778
## 203 0.8888889 0.1111111
## 204 0.5555556 0.4444444
## 205 0.8888889 0.1111111
## 206 0.3333333 0.6666667
## 207 0.2222222 0.7777778
## 208 0.0000000 1.0000000
## 209 0.2222222 0.7777778
## 210 0.3333333 0.6666667
## 211 0.5555556 0.4444444
## 212 0.4444444 0.5555556
## 213 0.5555556 0.4444444
## 214 0.4444444 0.5555556
## 215 0.2222222 0.7777778
## 216 0.3333333 0.6666667
## 217 0.0000000 1.0000000
## 218 0.4444444 0.5555556
## 219 0.4444444 0.5555556
## 220 0.4444444 0.5555556
## 221 0.8888889 0.1111111
## 222 0.4444444 0.5555556
## 223 0.1111111 0.8888889
## 224 0.5555556 0.4444444
## 225 0.3333333 0.6666667
## 226 0.3333333 0.6666667
## 227 0.2222222 0.7777778
## 228 0.2222222 0.7777778
## 229 0.6666667 0.3333333
## 230 0.0000000 1.0000000
## 231 0.0000000 1.0000000
## 232 0.4444444 0.5555556
## 233 0.3333333 0.6666667
## 234 0.8888889 0.1111111
## 235 0.7777778 0.2222222
## 236 0.3333333 0.6666667
## 237 0.4444444 0.5555556
## 238 0.2222222 0.7777778
## 239 0.2222222 0.7777778
## 240 0.8888889 0.1111111
## 241 0.2222222 0.7777778
## 242 0.4444444 0.5555556
## 243 0.5555556 0.4444444
## 244 0.1111111 0.8888889
## 245 0.0000000 1.0000000
## 246 0.1111111 0.8888889
## 247 0.8888889 0.1111111
## 248 0.5555556 0.4444444
## 249 0.4444444 0.5555556
## 250 0.5555556 0.4444444
## 251 0.5555556 0.4444444
## 252 0.4444444 0.5555556
## 253 0.4444444 0.5555556
## 254 0.2222222 0.7777778
## 255 0.4444444 0.5555556
## 256 0.0000000 1.0000000
## 257 0.3333333 0.6666667
## 258 0.3333333 0.6666667
## 259 0.6666667 0.3333333
## 260 0.3333333 0.6666667
## 261 0.5555556 0.4444444
## 262 0.7777778 0.2222222
## 263 0.2222222 0.7777778
```

``` r
roc_curve <- roc(heart_attack_dataset_test$class, predictions$neg)
```

```         
## Setting levels: control = negative, case = positive
```

```         
## Setting direction: controls > cases
```

``` r
# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)
```

![](Lab-Submission-Markdown_files/figure-html/code%20chunk%203-1.png)<!-- -->

# 4. Logarithmic Loss (LogLoss) ----

``` r
## 4.a. Load the dataset ----
## 4.b. Train the Model ----
# We apply the 5-fold repeated cross validation resampling method
# with 3 repeats
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)


heart_model_cart <- train(class ~ ., data = heart_attack_dataset, method = "rpart",
                         metric = "logLoss", trControl = train_control)

## 4.c. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----

print(heart_model_cart)
```

```         
## CART 
## 
## 1319 samples
##    8 predictor
##    2 classes: 'negative', 'positive' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 3 times) 
## Summary of sample sizes: 1056, 1055, 1055, 1055, 1055, 1055, ... 
## Resampling results across tuning parameters:
## 
##   cp           logLoss  
##   0.006876228  0.1637674
##   0.294695481  0.2193278
##   0.671905697  0.5320587
## 
## logLoss was used to select the optimal model using the smallest value.
## The final value used for the model was cp = 0.006876228.
```

# 5. References ----

``` r
## Kuhn, M., Wing, J., Weston, S., Williams, A., Keefer, C., Engelhardt, A., Cooper, T., Mayer, Z., Kenkel, B., R Core Team, Benesty, M., Lescarbeau, R., Ziem, A., Scrucca, L., Tang, Y., Candan, C., & Hunt, T. (2023). caret: Classification and Regression Training (6.0-94) [Computer software]. https://cran.r-project.org/package=caret # nolint ----

## Leisch, F., & Dimitriadou, E. (2023). mlbench: Machine Learning Benchmark Problems (2.1-3.1) [Computer software]. https://cran.r-project.org/web/packages/mlbench/index.html # nolint ----


## Robin, X., Turck, N., Hainard, A., Tiberti, N., Lisacek, F., Sanchez, J.-C., Müller, M., Siegert, S., Doering, M., & Billings, Z. (2023). pROC: Display and Analyze ROC Curves (1.18.4) [Computer software]. https://cran.r-project.org/web/packages/pROC/index.html # nolint ----

## Wickham, H., François, R., Henry, L., Müller, K., Vaughan, D., Software, P., & PBC. (2023). dplyr: A Grammar of Data Manipulation (1.1.3) [Computer software]. https://cran.r-project.org/package=dplyr # nolint ----

## Wickham, H., Chang, W., Henry, L., Pedersen, T. L., Takahashi, K., Wilke, C., Woo, K., Yutani, H., Dunnington, D., Posit, & PBC. (2023). ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics (3.4.3) [Computer software]. https://cran.r-project.org/package=ggplot2 # nolint ----
```
