## Code for Prediction with Machine Learning _ Assignment 1
# Set the working directory
setwd("D:/Onedrive-CEU/OneDrive - Central European University/CEU/Prediction with Machine Learning/Assignment/DA3-phdma/Assignment 1")

rm(list=ls())

setwd("D:/Onedrive-CEU/OneDrive - Central European University/CEU/Prediction with Machine Learning/Assignment/DA3-phdma/Assignment 1")

library(readr)
library(dplyr)
library(ggplot2)
library(weights)
library(caret)
library(kableExtra)
library(tidyr)


# Importing the data

DataOriginal <- read_csv("morg-2014-emp.csv")

# Filter - only select 'Miscellaneous agricultural workers, including animal breeders' - 6050

MiscWorkerData = DataOriginal %>%
  filter(occ2012 == 6050)

## Data Dictionary
MiscWorkerData %>% count(grade92)
MiscWorkerData %>% count(race)
MiscWorkerData %>% count(marital)
MiscWorkerData %>% count(prcitshp)
MiscWorkerData %>% count(class)

### Generated Variables
# Defining the target variable
MiscWorkerData = MiscWorkerData %>% mutate(earnhr = earnwke/uhours)

# grouping race. (1. white 2. black 3. non-white/black)
# Create is_white and is_black variables - use non-white/black as baseline

MiscWorkerData = MiscWorkerData %>% mutate(is_white = case_when(race == 1 ~ 1, race != 1 ~ 0), is_black = case_when(race == 2 ~ 1, race != 2 ~ 0))

# grouping marital statuses. (1. Married with spouse present 2. Married with spouse absent or seperated 3. Widowed or divorced 4. Never Married) - leave the first category out to be baseline

MiscWorkerData = MiscWorkerData %>% mutate(marr_abs = case_when(marital == 3 ~ 1, marital != 3 ~ 0))

MiscWorkerData = MiscWorkerData %>% mutate(wid_div = case_when(marital == 4 | marital == 5 | marital == 6 ~ 1, marital != 4 | marital != 5 | marital != 6 ~ 0))

MiscWorkerData = MiscWorkerData %>% mutate(nevmarr = case_when(marital == 7 ~ 1, marital != 7 ~ 0))

# Recode gender 1 = male, 0 = female. Use female as baseline
MiscWorkerData = MiscWorkerData %>% mutate(male = case_when(sex == 1 ~ 1, sex != 1 ~ 0))

# grouping citizen ship status. (1. Foreign Born, Not a US Citizen 2. Foreign Born, US Cit By Naturalization 3. Native) - use Native as baseline
MiscWorkerData = MiscWorkerData %>% mutate(noncitiz = case_when(prcitshp == 'Foreign Born, Not a US Citizen' ~ 1, prcitshp != 'Foreign Born, Not a US Citizen' ~ 0))

MiscWorkerData = MiscWorkerData %>% mutate(natura = case_when(prcitshp == 'Foreign Born, US Cit By Naturalization' ~ 1, prcitshp != 'Foreign Born, US Cit By Naturalization' ~ 0))

# grouping classes of worker (1. Private, For Profit, 2. Others) use Others as the baseline
MiscWorkerData = MiscWorkerData %>% mutate(forprofit = case_when(class == 'Private, For Profit' ~ 1, class != 'Private, For Profit' ~ 0))

# grouping education levels (1. 6th grade or less 2. 7th - 12th grade but NO Diploma, 3. High school graduate, diploma or GED 4. Some college but no degree 5. Associate degree 6. Bachelor's degree or more) - use 6th grade or less as baseline
MiscWorkerData = MiscWorkerData %>% mutate(than7nodip = case_when(between(grade92, 34, 38) ~ 1, !between(grade92, 34, 38) ~ 0))

MiscWorkerData = MiscWorkerData %>% mutate(HS_GED = case_when(grade92 == 39 ~ 1, grade92 != 39 ~ 0))

MiscWorkerData = MiscWorkerData %>% mutate(Col_ND = case_when(grade92 == 40 ~ 1, grade92 != 40 ~ 0))

MiscWorkerData = MiscWorkerData %>% mutate(asscd = case_when(between(grade92, 41, 42) ~ 1, !between(grade92, 41, 42) ~ 0))

MiscWorkerData = MiscWorkerData %>% mutate(Bach_more = case_when(grade92 >= 43 ~ 1, !grade92 >= 43 ~ 0))

### Model Building
## Added data about minimum wages
minwage = read_csv("Minwage_2014.csv")

# Left join into the main data
MiscWorkerData <- MiscWorkerData %>% left_join(minwage, by = c("stfips" = "State"))

## Selecting drop irrelevant variables
WorkingData = MiscWorkerData %>% select(-c(...1, hhid, intmonth, earnwke, race, ethnic, sex, marital, chldpres, prcitshp, state, occ2012, class, unionmme, unioncov, lfsr94, grade92, stfips))

# Calculating correlations matrix. (Only numeric variables)
# weighted correlations
data_mat = as.matrix(WorkingData[,-c(5)])

correlations = wtd.cors(x = data_mat[,-c(1)], y = data_mat[,5], weight = data_mat[,1])

# Put it into table
target_correlations <- data.frame(correlations)

# Print the correlations
print(target_correlations)

## Doing One-Hot Encoding to ind02 to get dummies
# Assuming your data is stored in a variable called 'data' and the nominal variable is 'nominal_variable'
dummy_data <- dummyVars(~ ind02, data = WorkingData)

# Apply the transformation to your data
data_encoded <- predict(dummy_data, newdata = WorkingData)

WorkingData = cbind(WorkingData, data_encoded)

## Doing 5-folds CV
set.seed(2023)
# Getting index for splitting data
folds <- createFolds(WorkingData$earnhr, k = 5, list = TRUE, returnTrain = FALSE)


## Estimating Models and obtains results (5-folds CV)
# Model 1
results <- list()
for (i in 1:5) {
  train_data <- WorkingData[-folds[[i]], ]
  test_data <- WorkingData[folds[[i]], ]
  
  # Train your OLS model
  model_1 <- lm(earnhr ~ age + marr_abs + wid_div + nevmarr + noncitiz + natura + than7nodip + HS_GED + Col_ND + asscd + Bach_more, weights = weight, data = train_data) 
  # Assuming 'target_variable' is the variable you're trying to predict, 
  # and you want to use all other variables as predictors
  
  # Test your model
  predictions <- predict(model_1, newdata = test_data)
  
  # Calculate performance metric MSE
  mse <- mean((test_data$earnhr - predictions)^2)
  
  # Store the results
  results[[i]] <- mse
}

# Get average RMSE
RMSE_CV_Model1 = sqrt(mean(unlist(results)))


# Model 2
results <- list()
for (i in 1:5) {
  train_data <- WorkingData[-folds[[i]], ]
  test_data <- WorkingData[folds[[i]], ]
  
  # Train your OLS model
  model_2 <- update(model_1, .~. + uhours, is_white, is_black, forprofit) 
  # Assuming 'target_variable' is the variable you're trying to predict, 
  # and you want to use all other variables as predictors
  
  # Test your model
  predictions <- predict(model_2, newdata = test_data)
  
  # Calculate performance metric MSE
  mse <- mean((test_data$earnhr - predictions)^2)
  
  # Store the results
  results[[i]] <- mse
}

# Get average RMSE
RMSE_CV_Model2 = sqrt(mean(unlist(results)))


# Model 3

results <- list()
for (i in 1:5) {
  train_data <- WorkingData[-folds[[i]], ]
  test_data <- WorkingData[folds[[i]], ]
  
  # Train your OLS model
  model_3 <- update(model_2, .~. + male, ownchild)
  # Assuming 'target_variable' is the variable you're trying to predict, 
  # and you want to use all other variables as predictors
  
  # Test your model
  predictions <- predict(model_3, newdata = test_data)
  
  # Calculate performance metric MSE
  mse <- mean((test_data$earnhr - predictions)^2)
  
  # Store the results
  results[[i]] <- mse
}

# Get average RMSE
RMSE_CV_Model3 = sqrt(mean(unlist(results)))


#Model 4
results <- list()
for (i in 1:5) {
  train_data <- WorkingData[-folds[[i]], ]
  test_data <- WorkingData[folds[[i]], ]
  
  # Train your OLS model
  model_4 <- lm(earnhr ~ . -weight, weights = weight, data = train_data[, -c(5)])
  # Assuming 'target_variable' is the variable you're trying to predict, 
  # and you want to use all other variables as predictors
  
  # Test your model
  predictions <- predict(model_4, newdata = test_data)
  
  # Calculate performance metric MSE
  mse <- mean((test_data$earnhr - predictions)^2)
  
  # Store the results
  results[[i]] <- mse
}

# Get average RMSE
RMSE_CV_Model4 = sqrt(mean(unlist(results)))

## Estimating using the whole sample. Get BIC and RMSE

## Model 1
model_1 <- lm(earnhr ~ age + marr_abs + wid_div + nevmarr + noncitiz + natura + than7nodip + HS_GED + Col_ND + asscd + Bach_more, weights = weight, data = WorkingData) 

# Calculate RMSE in full sample

predictions <- predict(model_1, newdata = WorkingData)

# Calculate performance metric RMSE
rmse_full_Model1 <- sqrt(mean((WorkingData$earnhr - predictions)^2))

# Calculate BIC
BIC_full_Model1 <- BIC(model_1)

## Model 2
model_2 <- update(model_1, .~. + uhours, is_white, is_black, forprofit) 

# Calculate RMSE in full sample

predictions <- predict(model_2, newdata = WorkingData)

# Calculate performance metric RMSE
rmse_full_Model2 <- sqrt(mean((WorkingData$earnhr - predictions)^2))

# Calculate BIC
BIC_full_Model2 <- BIC(model_2)

## Model 3
model_3 <- update(model_2, .~. + male, ownchild)

# Calculate RMSE in full sample

predictions <- predict(model_3, newdata = WorkingData)

# Calculate performance metric RMSE
rmse_full_Model3 <- sqrt(mean((WorkingData$earnhr - predictions)^2))

# Calculate BIC
BIC_full_Model3 <- BIC(model_3)

## Model 4
model_4 <- lm(earnhr ~ . -weight, weights = weight, data = WorkingData[, -c(5)])

# Calculate RMSE in full sample

predictions <- predict(model_4, newdata = WorkingData)

# Calculate performance metric RMSE
rmse_full_Model4 <- sqrt(mean((WorkingData$earnhr - predictions)^2))

# Calculate BIC
BIC_full_Model4 <- BIC(model_4)


### Performance
## Compile the performance indicators into a table
ColNames = c('RMSE in full sample', 'RMSE CV', 'BIC in full sample')

# Creating a dataframe
Sum_results <- data.frame(Column1 = c(rmse_full_Model1, rmse_full_Model2, rmse_full_Model3, rmse_full_Model4),
                          Column2 = c(RMSE_CV_Model1, RMSE_CV_Model2, RMSE_CV_Model3, RMSE_CV_Model4),
                          Column3 = c(BIC_full_Model1, BIC_full_Model2, BIC_full_Model3, BIC_full_Model4))


# Adding row names
rownames(Sum_results) <- c("Model 1", "Model 2", "Model 3","Model 4")

# Adding column names
colnames(Sum_results) <- ColNames

kable(Sum_results, align = "c")



## plot RMSE

# Convert it to long format
Sum_results = cbind(c("Model 1", "Model 2", "Model 3","Model 4") , Sum_results)
colnames(Sum_results)[1] <- 'Model'

data_long <- gather(Sum_results, key = "Metric", value = "Value", -Model)

# Filter the data to include only "RMSE in full sample" and "RMSE CV"
df_plot <- data_long[data_long$Metric %in% c("RMSE in full sample", "RMSE CV"), ]

# Create a line plot
ggplot(df_plot, aes(x = Model, y = Value, color = Metric, group = Metric)) +
  geom_line() +
  geom_point() +
  labs(title = "RMSE in Full Sample and RMSE CV for Each Model",
       x = "Model",
       y = "Value") +
  scale_color_manual(values = c("RMSE in full sample" = "blue", "RMSE CV" = "red")) +   scale_y_continuous(limits = c(2, 8)) +  # Set y-axis limits + 
  theme_minimal() + theme(plot.title = element_text(size = 20), axis.text = element_text(size = 10))  # Adjust the title font size

