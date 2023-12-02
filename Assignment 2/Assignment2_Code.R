rm(list=ls())

library(readr)
library(dplyr)
library(ggplot2)
library(weights)
library(caret)
library(kableExtra)
library(tidyr)
library(fastDummies)
library(data.table)
library(car)
library(ipred)
library(rpart) 
library(randomForest)
library(treeshap)
library(shapviz)

# Set Working Directory and Importing Data

setwd("D:/Onedrive-CEU/OneDrive - Central European University/CEU/Prediction with Machine Learning/Assignment/DA3-phdma/Assignment 2")

Data_Mar <- read_csv("listings_mar_2023.csv", show_col_types = FALSE, na = c("", "NA", "N/A"))

##### Task 1
### Defining Live Data and Sample

### Filtering data
Data_Mar = Data_Mar %>% 
  filter(property_type %in% c("Entire serviced apartment", "Private room in serviced apartment", "Room in serviced apartment", "Shared room in serviced apartment", "Entire home/apt", "Entire condo", "Private room in condo", "Shared room in condo"), accommodates >= 2, accommodates <= 6)

### Split sample into hold-out and work set.
set.seed(20231127)

## hold-out set will only consists of apartments. Split 20% of apartments into hold-out set.
hold_out_Mar <- Data_Mar %>%
  filter(property_type %in% c("Entire serviced apartment", "Private room in serviced apartment", "Room in serviced apartment", "Shared room in serviced apartment")) %>%
  sample_frac(0.20)

## Subtract hold-out set from the sample to get work set.
Data_Mar <- Data_Mar %>%
  filter(!id %in% hold_out_Mar$id)

## Deleting id column
hold_out_Mar <- hold_out_Mar %>%
  select(-id)

Data_Mar <- Data_Mar %>%
  select(-id)

### Converting the price to numeric 
Data_Mar$price <- gsub("\\$", "", Data_Mar$price)
Data_Mar$price <- as.numeric(gsub(",", "", Data_Mar$price))

hold_out_Mar$price <- gsub("\\$", "", hold_out_Mar$price)
hold_out_Mar$price <- as.numeric(gsub(",", "", hold_out_Mar$price))


### Feature Engineering
### Keeping only variables that we are going to use.
Data_Mar = Data_Mar %>% 
  select(price, host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost, host_total_listings_count, host_identity_verified, neighbourhood_cleansed, property_type, room_type, accommodates, bathrooms_text, bedrooms, beds, amenities, minimum_nights, maximum_nights, has_availability, availability_30, availability_60, availability_90, availability_365, number_of_reviews, last_review, last_scraped,review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, instant_bookable, description, neighborhood_overview)


## Converting host_response_rate to numeric
Data_Mar$host_response_rate <- gsub("\\%", "", Data_Mar$host_response_rate)

Data_Mar$host_response_rate <- as.numeric(gsub(",", "", Data_Mar$host_response_rate))

## Converting host_acceptance_rate to numeric
Data_Mar$host_acceptance_rate <- gsub("\\%", "", Data_Mar$host_acceptance_rate)

Data_Mar$host_acceptance_rate <- as.numeric(gsub(",", "", Data_Mar$host_acceptance_rate))

## Let's tranform bathrooms_text.
Data_Mar %>% 
  count(bathrooms_text)

## It seems that we have 3 main categories
# 1. baths 2. shared baths 3.private baths 4. Half-bath
# Recode it in terms of number of baths (each category)


### Extracting number and types of bathrooms
## Start with March
# Initialize new columns
Data_Mar$num_baths <- numeric(nrow(Data_Mar))
Data_Mar$num_shared_baths <- numeric(nrow(Data_Mar))
Data_Mar$num_private_baths <- numeric(nrow(Data_Mar))
Data_Mar$is_half_bath <- numeric(nrow(Data_Mar))

# Extract information using regular expressions
for (i in seq_along(Data_Mar$bathrooms_text)) {
  # Check for NA
  if (!is.na(Data_Mar$bathrooms_text[i])) {
    # Extract number and type of bath
    match <- regmatches(Data_Mar$bathrooms_text[i], regexec("(\\d+)\\s*(shared|private)?\\s*baths?", Data_Mar$bathrooms_text[i]))[[1]]
    
    # Extract number
    num <- as.numeric(match[2])
    
    # Determine type and assign to corresponding variable
    if (grepl("shared", match[3], ignore.case = TRUE)) {
      Data_Mar$num_shared_baths[i] <- num
    } else if (grepl("private", match[3], ignore.case = TRUE)) {
      Data_Mar$num_private_baths[i] <- num
    } else {
      Data_Mar$num_baths[i] <- num
    }
    
    # Check if it is a half bath
    Data_Mar$is_half_bath[i] <- ifelse(grepl("half", tolower(Data_Mar$bathrooms_text[i])) || grepl("shared half", tolower(Data_Mar$bathrooms_text[i])), 1, 0)
  }
}

# Set values to NA if bathrooms_text is NA
Data_Mar$num_baths[is.na(Data_Mar$bathrooms_text)] <- NA
Data_Mar$num_shared_baths[is.na(Data_Mar$bathrooms_text)] <- NA
Data_Mar$num_private_baths[is.na(Data_Mar$bathrooms_text)] <- NA
Data_Mar$is_half_bath[is.na(Data_Mar$bathrooms_text)] <- NA


# Drop the original 'bathrooms_text' column
Data_Mar <- subset(Data_Mar, select = -bathrooms_text)

### Extracting amenities into dummy variables
# List of amenities
amenities_list <- c("Hair dryer", "Shampoo", "Shower gel", "Air conditioning", "Essentials", 
                    "Wifi", "Washer", "Iron", "Smoking allowed", "Free parking on premises", 
                    "Luggage dropoff allowed", "Kitchen", "Refrigerator", "Dining table", 
                    "Dedicated workspace", "Elevator", "Microwave", "Dishes and silverware", 
                    "TV", "Pool", "Gym", "Pets allowed")

# Loop through the list and create dummy variables
for (amenity in amenities_list) {
  dummy_variable_name <- paste0(gsub(" ", "_", amenity), "_dummy")
  Data_Mar[[dummy_variable_name]] <- ifelse(sapply(Data_Mar$amenities, grepl, pattern = amenity), 1, 0)
}


# Drop the original 'amenities' column
Data_Mar <- subset(Data_Mar, select = -amenities)


### Encoding other variables into dummy variables
## host_response_time
Data_Mar = dummy_cols(Data_Mar, select_columns = "host_response_time", remove_first_dummy = TRUE, remove_selected_columns = TRUE)


# drop host_response_time_NA
Data_Mar <- subset(Data_Mar, select = -host_response_time_NA)


## neighbourhood_cleansed
Data_Mar = dummy_cols(Data_Mar, select_columns = "neighbourhood_cleansed", remove_first_dummy = TRUE, remove_selected_columns = TRUE)

## property_type
Data_Mar = dummy_cols(Data_Mar, select_columns = "property_type", remove_first_dummy = TRUE, remove_selected_columns = TRUE)


## room_type
Data_Mar = dummy_cols(Data_Mar, select_columns = "room_type", remove_first_dummy = TRUE, remove_selected_columns = TRUE)



### Creating variable called "time_since_last_review"
# Convert data to date first

Data_Mar$last_review <- as.Date(Data_Mar$last_review, "%Y-%b-%d")

Data_Mar$last_scraped <- as.Date(Data_Mar$last_scraped, "%Y-%b-%d")


# Calculate the time between the two dates

Data_Mar$time_since_last_review <- Data_Mar$last_scraped - Data_Mar$last_review


# Removing the used two columns
Data_Mar = Data_Mar %>%
  select(-c(last_review, last_scraped))



### Extracting Information about Skytrain, Metro
## MRT (Subway)

# Define a vector of keywords related to public transport
MRT_keywords <- c("\\bMRT\\b", "\\bmrt\\b", "\\bMetro\\b", "\\bmetro\\b", "\\bSubway\\b", "\\bsubway\\b")

# Use the keywords in grepl and create a new column
Data_Mar$near_MRT <- ifelse(
  grepl(paste(MRT_keywords, collapse = "|"), 
        Data_Mar$description, ignore.case = TRUE) |
    grepl(paste(MRT_keywords, collapse = "|"), 
          Data_Mar$neighborhood_overview, ignore.case = TRUE),
  1, 0
)

## BTS (Skytrain)
BTS_keywords <- c("\\bBTS\\b", "\\bbts\\b", "\\bSkytrain\\b", "\\bskytrain\\b")

Data_Mar$near_BTS <- ifelse(
  grepl(paste(BTS_keywords, collapse = "|"), 
        Data_Mar$description, ignore.case = TRUE) |
    grepl(paste(BTS_keywords, collapse = "|"), 
          Data_Mar$neighborhood_overview, ignore.case = TRUE),
  1, 0
)


## ARL (Airport Rail Link)
ARL_keywords <- c("\\bAirport Rail Link\\b", "\\bAirport Link\\b", "\\bairport rail link\\b", "\\bairport link\\b", "\\bARL\\b", "\\barl\\b")

Data_Mar$near_ARL <- ifelse(
  grepl(paste(ARL_keywords, collapse = "|"), 
        Data_Mar$description, ignore.case = TRUE) |
    grepl(paste(ARL_keywords, collapse = "|"), 
          Data_Mar$neighborhood_overview, ignore.case = TRUE),
  1, 0
)

# Set values to NA if description and neighborhood_overview are NA

Data_Mar <- Data_Mar %>%
  mutate(near_MRT = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_MRT))

Data_Mar <- Data_Mar %>%
  mutate(near_BTS = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_BTS))

Data_Mar <- Data_Mar %>%
  mutate(near_ARL = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_ARL))

# Drop description and neighborhood_overview
Data_Mar = Data_Mar %>%
  select(-c(description, neighborhood_overview))


### Dealing with NAs
## Examining NA values

NA_Mar <- Data_Mar %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "na_count") %>%
  arrange(desc(na_count)) 

kable(head(NA_Mar, 20))

## Drop those with missing values for host_acceptance_rate and host_response_rate

Data_Mar <- Data_Mar %>%
  filter(!is.na(host_acceptance_rate) & !is.na(host_response_rate))


## Drop those with missing values for baths
Data_Mar <- Data_Mar %>%
  filter(!is.na(num_baths) & !is.na(num_shared_baths) & !is.na(is_half_bath) & !is.na(num_private_baths))

## Replace NA values in near_MRT, near_BTS, and near_ARL with 0

Data_Mar <- Data_Mar %>%
  mutate(across(c(near_MRT, near_BTS, near_ARL), ~ ifelse(is.na(.), 0, .)))



## Replace NA values in host_is_superhost with 0
Data_Mar <- Data_Mar %>%
  mutate(host_is_superhost = ifelse(is.na(host_is_superhost), 0, host_is_superhost))

## Replace NA values in time_since_last_review with max
Data_Mar <- Data_Mar %>%
  mutate(time_since_last_review = ifelse(is.na(time_since_last_review), max(time_since_last_review, na.rm = TRUE), time_since_last_review))

## Replace NA values in scores with mean
Data_Mar <- Data_Mar %>%
  mutate(across(c(review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))


# Create a binary flag variable to capture the imputation
Data_Mar <- Data_Mar %>%
  mutate(review_scores_rating_imputed = ifelse(is.na(review_scores_rating), 1, 0))

## Replace NA values in bedrooms and beds with mean
Data_Mar <- Data_Mar %>%
  mutate(across(c(bedrooms, beds), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))


# Create a binary flag variable to capture the imputation
Data_Mar <- Data_Mar %>%
  mutate(bedrooms_imputed = ifelse(is.na(bedrooms), 1, 0))


### Building Models
## Linear Regression

selected_variables <- c("review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value")

# Compute the correlation matrix for selected variables
cor(Data_Mar[, selected_variables], use = "complete.obs")

## Compute the correlation between price and the important predictors.

# Compute the correlation matrix for selected variables
cor_main <- cor(Data_Mar[, c("price", "accommodates", "bedrooms", "beds", "num_baths", "num_shared_baths", "num_private_baths", "near_ARL", "near_MRT", "near_BTS", "room_type_Private room", "room_type_Shared room")], use = "complete.obs")

# Extract the correlation between price and the important predictors
kable(cor_main[1,], align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))


### Selecting by using 5-fold cross validation
set.seed(20231127)
# Getting the indices for 5-fold cross validation
folds <- createFolds(Data_Mar$price, k = 5, list = TRUE, returnTrain = TRUE)

### Estimating Models and obtains RMSE (5-folds CV)
## Model 1
results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_1 <- lm(price ~ accommodates + bedrooms + beds + bedrooms_imputed, data = train)
  
  
  # Predict the test data
  test$pred <- predict(model_OLS_1, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_1 <- sqrt(mean(unlist(results)^2))


## Model 2
results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_2 <- update(model_OLS_1, . ~ . + num_baths + num_shared_baths + is_half_bath + num_private_baths + `property_type_Entire home/apt` + `property_type_Entire serviced apartment` + `property_type_Private room in condo` + `property_type_Private room in serviced apartment` + `property_type_Room in serviced apartment` + `property_type_Shared room in serviced apartment` + `property_type_Shared room in condo` + `room_type_Hotel room` + `room_type_Private room` + `room_type_Shared room`, data = train)
  
  
  # Predict the test data
  test$pred <- predict(model_OLS_2, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_2 <- sqrt(mean(unlist(results)^2))


## Model 3
results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_3 <- update(model_OLS_2, . ~ . + near_MRT + near_BTS + near_ARL, data = train)
  
  # Predict the test data
  test$pred <- predict(model_OLS_3, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_3 <- sqrt(mean(unlist(results)^2))


## Model 4
results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_4 <- update(model_OLS_3, . ~ . + review_scores_rating + review_scores_accuracy + review_scores_cleanliness + review_scores_checkin + review_scores_communication + review_scores_location + review_scores_value + review_scores_rating_imputed + time_since_last_review + number_of_reviews, data = train)
  
  
  # Predict the test data
  test$pred <- predict(model_OLS_4, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_4 <- sqrt(mean(unlist(results)^2))


## Model 5
results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_5 <- update(model_OLS_4, . ~ . +  I(accommodates^2) + I(bedrooms^2) + I(beds^2), data = train)
  
  
  # Predict the test data
  test$pred <- predict(model_OLS_5, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_5 <- sqrt(mean(unlist(results)^2))


## Model 6
results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_6 <- update(model_OLS_5, . ~ . + `neighbourhood_cleansed_Bang Kapi` + `neighbourhood_cleansed_Bang Khae` + `neighbourhood_cleansed_Bang Khen` + `neighbourhood_cleansed_Bang Kho laen` + `neighbourhood_cleansed_Bang Khun thain` + `neighbourhood_cleansed_Bang Na` + `neighbourhood_cleansed_Bang Phlat` + `neighbourhood_cleansed_Bang Rak` + `neighbourhood_cleansed_Bang Sue` + `neighbourhood_cleansed_Bangkok Noi` + `neighbourhood_cleansed_Bangkok Yai` + `neighbourhood_cleansed_Bueng Kum` + `neighbourhood_cleansed_Chatu Chak` + `neighbourhood_cleansed_Chom Thong` + `neighbourhood_cleansed_Din Daeng` + `neighbourhood_cleansed_Don Mueang` + `neighbourhood_cleansed_Dusit` + `neighbourhood_cleansed_Huai Khwang` + `neighbourhood_cleansed_Khan Na Yao` +  `neighbourhood_cleansed_Khlong San` + `neighbourhood_cleansed_Khlong Toei` + `neighbourhood_cleansed_Lak Si` + `neighbourhood_cleansed_Lat Krabang` + `neighbourhood_cleansed_Lat Phrao` + `neighbourhood_cleansed_Min Buri` +  `neighbourhood_cleansed_Nong Khaem` + `neighbourhood_cleansed_Phasi Charoen` +   `neighbourhood_cleansed_Ratchathewi` + `neighbourhood_cleansed_Sai Mai` + `neighbourhood_cleansed_Samphanthawong` + `neighbourhood_cleansed_Saphan Sung` + `neighbourhood_cleansed_Sathon` + `neighbourhood_cleansed_Taling Chan` +  `neighbourhood_cleansed_Thon buri` +  `neighbourhood_cleansed_Parthum Wan` + `neighbourhood_cleansed_Phaya Thai` + `neighbourhood_cleansed_Phra Khanong` + `neighbourhood_cleansed_Phra Nakhon` + `neighbourhood_cleansed_Pom Prap Sattru Phai` + `neighbourhood_cleansed_Pra Wet` + `neighbourhood_cleansed_Rat Burana` +      `neighbourhood_cleansed_Suanluang` + `neighbourhood_cleansed_Thung khru` + `neighbourhood_cleansed_Vadhana` + `neighbourhood_cleansed_Wang Thong Lang` + `neighbourhood_cleansed_Yan na wa`, data = train)
  
  # Predict the test data
  test$pred <- predict(model_OLS_6, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_6 <- sqrt(mean(unlist(results)^2))

## Model 7
results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_7 <- update(model_OLS_6, . ~ . + host_is_superhost + host_response_rate + host_acceptance_rate + host_total_listings_count + host_identity_verified + `host_response_time_within a few hours` + `host_response_time_within an hour` + `host_response_time_within a day`, data = train)
  
  # Predict the test data
  test$pred <- predict(model_OLS_7, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_7 <- sqrt(mean(unlist(results)^2))


## Model 8

results <- list()

for (i in 1:5) {
  # Get the training and test data
  train <- Data_Mar[folds[[i]], ]
  test <- Data_Mar[-folds[[i]], ]
  
  # Estimate the model
  model_OLS_8 <- lm(price ~ ., data = train)
  
  # Predict the test data
  test$pred <- predict(model_OLS_8, newdata = test)
  
  # Calculate the RMSE
  rmse <- RMSE(test$pred, test$price)
  
  # Store the results
  results[[i]] <- rmse
}

# Calculate the average RMSE
RMSE_OLS_8 <- sqrt(mean(unlist(results)^2))


## Put the results in a data frame
results <- data.frame(
  Model = c("OLS_1", "OLS_2", "OLS_3", "OLS_4", "OLS_5", "OLS_6", "OLS_7", "OLS_8"),
  RMSE = c(RMSE_OLS_1, RMSE_OLS_2, RMSE_OLS_3, RMSE_OLS_4, RMSE_OLS_5, RMSE_OLS_6, RMSE_OLS_7, RMSE_OLS_8))

# Print the table with borders
kable(results, digits = 2, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))


## Re-esimate the model with the optimal set of variables (Model 2)

model_OLS_optm <- update(model_OLS_1, . ~ . + num_baths + num_shared_baths + is_half_bath + num_private_baths + `property_type_Entire home/apt` + `property_type_Entire serviced apartment` + `property_type_Private room in condo` + `property_type_Private room in serviced apartment` + `property_type_Room in serviced apartment` + `property_type_Shared room in serviced apartment` + `property_type_Shared room in condo` + `room_type_Hotel room` + `room_type_Private room` + `room_type_Shared room`, data = Data_Mar)


## Random Forest Model
## Setting up the random forest
# specifying the range of values for the tuning parameters

mtry <- seq(20, 40, by = 1)

# Setting up the grid
grid <- expand.grid(mtry = mtry)

# setting up the control
control <- trainControl(method = "cv", number = 5)

# tuning the model
set.seed(20231128)

# I set ntree = 200 for the sake of computation time.
model_RF <- train(price ~ ., data = Data_Mar, method = "rf", trControl = control, tuneGrid = grid, ntree = 200)

## Display the results of Random Forest.
model_RF

## Re-estimate the model with the optimal paremeters (mtry = model_RF$bestTune$mtry)
set.seed(20231127)

predictors = subset(Data_Mar, select = -c(price))
model_RF_optm <- randomForest(x = predictors, y = Data_Mar$price, ntree = 200, mtry = model_RF$bestTune$mtry)

### Bagging Model
### Constructing the bagging model using ipred package. Choose the optimal stopping rule (minsplit) using 5-fold cross-validation.

# Setting up the range of values for the tuning parameters
minsplit_range <- c(2, 4, 6, 8, 10)

# Selecting by using 5-fold cross validation
set.seed(20231127)
# Getting the indices for 5-fold cross validation
folds <- createFolds(Data_Mar$price, k = 5, list = TRUE, returnTrain = TRUE)

# Set up the vector to collect the results from each model
RMSE_Bagging <- c()

for (parameter in 1:length(minsplit_range)) {
  results <- list()
  for (i in 1:5) {
    # Get the training and test data
    train <- Data_Mar[folds[[i]], ]
    test <- Data_Mar[-folds[[i]], ]
    
    # Estimate the model
    model_bagging <- bagging(price ~ ., data = train, nbagg = 200, control = rpart.control(minsplit = minsplit_range[parameter]))
    
    # Predict the test data
    test$pred <- predict(model_bagging, newdata = test)
    
    # Calculate the RMSE
    rmse <- RMSE(test$pred, test$price)
    
    # Store the results
    results[[i]] <- rmse
    
    # average across the 5 folds
    avg_rmse <- sqrt(mean(unlist(results)^2))
  }
  RMSE_Bagging[parameter] <- avg_rmse
}


## Combine the results into a data frame
results <- data.frame(
  minsplit = minsplit_range,
  RMSE = RMSE_Bagging
)

# Print it with Kable
kable(results, digits = 2, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))


#The optimal stopping rule chosen by CV is minsplit = 6, since it is the model with the lowest RMSE.

## Re-estimate the model with the optimal stopping rule (minsplit = minsplit_optm)

## Determine the optimal stopping rule
minsplit_optm <- results[which.min(results$RMSE), "minsplit"]

set.seed(20231127)
model_bagging_optm <- bagging(price ~ ., data = Data_Mar, nbagg = 200, control = rpart.control(minsplit = minsplit_optm))

### Evaluating Performance
### Cleaning the holdout data with the same process.
# Keep the columns that are used in the model
hold_out_Mar = hold_out_Mar %>% 
  select(price, host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost, host_total_listings_count, host_identity_verified, neighbourhood_cleansed, property_type, room_type, accommodates, bathrooms_text, bedrooms, beds, amenities, minimum_nights, maximum_nights, has_availability, availability_30, availability_60, availability_90, availability_365, number_of_reviews, last_review, last_scraped, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, instant_bookable, description, neighborhood_overview)


# Convert the host_response_rate column to numeric
hold_out_Mar$host_response_rate <- gsub("\\%", "", hold_out_Mar$host_response_rate)

hold_out_Mar$host_response_rate <- as.numeric(gsub(",", "", hold_out_Mar$host_response_rate))

# Convert the host_acceptance_rate column to numeric
hold_out_Mar$host_acceptance_rate <- gsub("\\%", "", hold_out_Mar$host_acceptance_rate)

hold_out_Mar$host_acceptance_rate <- as.numeric(gsub(",", "", hold_out_Mar$host_acceptance_rate))

# Initialize new columns
hold_out_Mar$num_baths <- numeric(nrow(hold_out_Mar))
hold_out_Mar$num_shared_baths <- numeric(nrow(hold_out_Mar))
hold_out_Mar$num_private_baths <- numeric(nrow(hold_out_Mar))
hold_out_Mar$is_half_bath <- numeric(nrow(hold_out_Mar))

# Extract information using regular expressions
for (i in seq_along(hold_out_Mar$bathrooms_text)) {
  # Check for NA
  if (!is.na(hold_out_Mar$bathrooms_text[i])) {
    # Extract number and type of bath
    match <- regmatches(hold_out_Mar$bathrooms_text[i], regexec("(\\d+)\\s*(shared|private)?\\s*baths?", hold_out_Mar$bathrooms_text[i]))[[1]]
    
    # Extract number
    num <- as.numeric(match[2])
    
    # Determine type and assign to corresponding variable
    if (grepl("shared", match[3], ignore.case = TRUE)) {
      hold_out_Mar$num_shared_baths[i] <- num
    } else if (grepl("private", match[3], ignore.case = TRUE)) {
      hold_out_Mar$num_private_baths[i] <- num
    } else {
      hold_out_Mar$num_baths[i] <- num
    }
    
    # Check if it is a half bath
    hold_out_Mar$is_half_bath[i] <- ifelse(grepl("half", tolower(hold_out_Mar$bathrooms_text[i])) || grepl("shared half", tolower(hold_out_Mar$bathrooms_text[i])), 1, 0)
  }
}

# Set values to NA if bathrooms_text is NA
hold_out_Mar$num_baths[is.na(hold_out_Mar$bathrooms_text)] <- NA
hold_out_Mar$num_shared_baths[is.na(hold_out_Mar$bathrooms_text)] <- NA
hold_out_Mar$num_private_baths[is.na(hold_out_Mar$bathrooms_text)] <- NA
hold_out_Mar$is_half_bath[is.na(hold_out_Mar$bathrooms_text)] <- NA

# Drop the original 'bathrooms_text' column
hold_out_Mar <- subset(hold_out_Mar, select = -bathrooms_text)

## Extracting amenities into dummy variables
for (amenity in amenities_list) {
  dummy_variable_name <- paste0(gsub(" ", "_", amenity), "_dummy")
  hold_out_Mar[[dummy_variable_name]] <- ifelse(sapply(hold_out_Mar$amenities, grepl, pattern = amenity), 1, 0)
}

# Drop the original 'amenities' column
hold_out_Mar <- subset(hold_out_Mar, select = -amenities)

## Encoding other variables into dummy variables
# host_response_time
hold_out_Mar = dummy_cols(hold_out_Mar, select_columns = "host_response_time", remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# drop host_response_time_NA
hold_out_Mar <- subset(hold_out_Mar, select = -host_response_time_NA)

# neighbourhood_cleansed
hold_out_Mar = dummy_cols(hold_out_Mar, select_columns = "neighbourhood_cleansed", remove_first_dummy = TRUE, remove_selected_columns = TRUE)


# property_type
hold_out_Mar = dummy_cols(hold_out_Mar, select_columns = "property_type", remove_first_dummy = TRUE)

# room_type
hold_out_Mar = dummy_cols(hold_out_Mar, select_columns = "room_type", remove_first_dummy = TRUE, remove_selected_columns = TRUE)


## Creating variable called "time_since_last_review"
# Convert data to date first
hold_out_Mar$last_review <- as.Date(hold_out_Mar$last_review, "%Y-%b-%d")

hold_out_Mar$last_scraped <- as.Date(hold_out_Mar$last_scraped, "%Y-%b-%d")

# Calculate the time between the two dates
hold_out_Mar$time_since_last_review <- hold_out_Mar$last_scraped - hold_out_Mar$last_review

# Removing the used two columns
hold_out_Mar = hold_out_Mar %>%
  select(-c(last_review, last_scraped))


## Extrating Information about Skytrain, Metro
hold_out_Mar$near_MRT <- ifelse(
  grepl(paste(MRT_keywords, collapse = "|"), 
        hold_out_Mar$description, ignore.case = TRUE) |
    grepl(paste(MRT_keywords, collapse = "|"), 
          hold_out_Mar$neighborhood_overview, ignore.case = TRUE),
  1, 0
)

hold_out_Mar$near_BTS <- ifelse(
  grepl(paste(BTS_keywords, collapse = "|"), 
        hold_out_Mar$description, ignore.case = TRUE) |
    grepl(paste(BTS_keywords, collapse = "|"), 
          hold_out_Mar$neighborhood_overview, ignore.case = TRUE),
  1, 0
)

hold_out_Mar$near_ARL <- ifelse(
  grepl(paste(ARL_keywords, collapse = "|"), 
        hold_out_Mar$description, ignore.case = TRUE) |
    grepl(paste(ARL_keywords, collapse = "|"), 
          hold_out_Mar$neighborhood_overview, ignore.case = TRUE),
  1, 0
)


hold_out_Mar <- hold_out_Mar %>%
  mutate(near_MRT = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_MRT))

hold_out_Mar <- hold_out_Mar %>%
  mutate(near_BTS = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_BTS))

hold_out_Mar <- hold_out_Mar %>%
  mutate(near_ARL = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_ARL))

hold_out_Mar = hold_out_Mar %>% select(-c(description, neighborhood_overview))

## Dealing with NAs
# Drop those with missing values for host_acceptance_rate and host_response_rate
hold_out_Mar <- hold_out_Mar %>% 
  filter(!is.na(host_acceptance_rate) & !is.na(host_response_rate))

# Drop those with missing values for baths
hold_out_Mar <- hold_out_Mar %>% 
  filter(!is.na(num_baths) & !is.na(num_shared_baths) & !is.na(is_half_bath) & !is.na(num_private_baths))

# Replace NA values in near_MRT, near_BTS, and near_ARL with 0

hold_out_Mar <- hold_out_Mar %>% 
  mutate(across(c(near_MRT, near_BTS, near_ARL), ~ ifelse(is.na(.), 0, .)))

# Replace NA values in host_is_superhost with 0
hold_out_Mar <- hold_out_Mar %>% 
  mutate(host_is_superhost = ifelse(is.na(host_is_superhost), 0, host_is_superhost))

# Replace NA values in time_since_last_review with max
hold_out_Mar <- hold_out_Mar %>% 
  mutate(time_since_last_review = ifelse(is.na(time_since_last_review), max(time_since_last_review, na.rm = TRUE), time_since_last_review))

## Replace NA values in scores with mean
hold_out_Mar <- hold_out_Mar %>% 
  mutate(across(c(review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Create a binary flag variable to capture the imputation
hold_out_Mar <- hold_out_Mar %>% 
  mutate(review_scores_rating_imputed = ifelse(is.na(review_scores_rating), 1, 0))

## Replace NA values in bedrooms and beds with mean
hold_out_Mar <- hold_out_Mar %>% 
  mutate(across(c(bedrooms, beds), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Create a binary flag variable to capture the imputation
hold_out_Mar <- hold_out_Mar %>% 
  mutate(bedrooms_imputed = ifelse(is.na(bedrooms), 1, 0))

## Keeping only apartments
hold_out_Mar = hold_out_Mar %>% 
  filter(property_type %in% c("Entire serviced apartment", "Private room in serviced apartment", "Room in serviced apartment", "Shared room in serviced apartment"))

# Drop property_type
hold_out_Mar <- subset(hold_out_Mar, select = -property_type)

### The work set and the holdout set will have different columns because of the size of the holdout set. There might not be observations from some neighbourhooods or observations with certain amenities in the holdout set. Therefore, we need to make sure that the work set and the holdout set have the same columns. 

## Create a list of columns that are in the work set but not in the holdout set
cols_to_add_holdout <- setdiff(colnames(Data_Mar), colnames(hold_out_Mar))

## Add the columns to the holdout set
hold_out_Mar[cols_to_add_holdout] <- 0


## Examining NA values again for the holdout set

NA_hold_out <- hold_out_Mar %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "na_count") %>%
  arrange(desc(na_count)) 

kable(head(NA_hold_out, 5))


## RMSE Results
## Use OLS model 2 to predict the price and calculate the RMSE in the holdout set.

# Use OLS

hold_out_Mar$price_pred_ols <- predict(model_OLS_optm, newdata = hold_out_Mar)

# Calculate the RMSE
RMSE_OLS_Holdout <- RMSE(hold_out_Mar$price, hold_out_Mar$price_pred_ols)

# Calculate the RMSE/Mean(Price)
'RMSE_OLS/Mean(Price)' <- RMSE_OLS_Holdout/mean(hold_out_Mar$price)

## Use the chosen random forest model to predict the price and calculate the RMSE in the holdout set.

hold_out_Mar$price_pred_RF <- predict(model_RF_optm, newdata = hold_out_Mar)

# Calculate the RMSE
RMSE_RF_Holdout <- RMSE(hold_out_Mar$price, hold_out_Mar$price_pred_RF)

# Calculate the RMSE/Mean(Price)
'RMSE_RF/Mean(Price)' <- RMSE_RF_Holdout/mean(hold_out_Mar$price)


## Use the chosen random forest model to predict the price and calculate the RMSE in the holdout set.

hold_out_Mar$price_pred_RF <- predict(model_RF_optm, newdata = hold_out_Mar)

# Calculate the RMSE
RMSE_RF_Holdout <- RMSE(hold_out_Mar$price, hold_out_Mar$price_pred_RF)

# Calculate the RMSE/Mean(Price)
'RMSE_RF/Mean(Price)' <- RMSE_RF_Holdout/mean(hold_out_Mar$price)


## Use the estimated bagging model with minsplit = 6 to predict the price and calculate the RMSE in the holdout set.

hold_out_Mar$price_pred_bag <- predict(model_bagging_optm, newdata = hold_out_Mar)

# Calculate the RMSE
RMSE_bagging_optm_Holdout <- RMSE(hold_out_Mar$price, hold_out_Mar$price_pred_bag)

# Calculate the RMSE/Mean(Price)
'RMSE_Bagging/Mean(Price)' <- RMSE_bagging_optm_Holdout/mean(hold_out_Mar$price)


## Compile the RMSE results and RMSE/Mean(Price) results for the holdout set

RMSE_Holdout <- data.frame(RMSE_OLS_Holdout, RMSE_RF_Holdout, RMSE_bagging_optm_Holdout) %>%
  pivot_longer(everything(), names_to = "model", values_to = "RMSE") %>%
  mutate(model = c("OLS", "RF", "Bagging"))

# add column for RMSE/Mean(Price)
RMSE_Holdout$'RMSE/Price' <- c(`RMSE_OLS/Mean(Price)`, `RMSE_RF/Mean(Price)`, `RMSE_Bagging/Mean(Price)`)


## Compare the performance across apartment size (small apartments: M= 3, large apartments > 3)

## Calculate the Mean Price for small apartments and large apartments in the holdout set.

Mean_price_small <- hold_out_Mar %>% 
  filter(accommodates <= 3) %>% 
  summarise(Mean_price_small = mean(price))

Mean_price_large <- hold_out_Mar %>% 
  filter(accommodates > 3) %>% 
  summarise(Mean_price_large = mean(price))

## For each model, calculate the RMSE for small apartments and large apartments in the holdout set using the RMSE function.

# OLS
RMSE_OLS_small <- RMSE(as.data.frame(hold_out_Mar %>% filter(accommodates <= 3) %>% select(price))$price, as.data.frame(hold_out_Mar %>% filter(accommodates <= 3) %>% select(price_pred_ols))$price_pred_ols)

RMSE_OLS_large <- RMSE(as.data.frame(hold_out_Mar %>% filter(accommodates > 3) %>% select(price))$price, as.data.frame(hold_out_Mar %>% filter(accommodates > 3) %>% select(price_pred_ols))$price_pred_ols)

# RF
RMSE_RF_small <- RMSE(as.data.frame(hold_out_Mar %>% filter(accommodates <= 3) %>% select(price))$price, as.data.frame(hold_out_Mar %>% filter(accommodates <= 3) %>% select(price_pred_RF))$price_pred_RF)

RMSE_RF_large <- RMSE(as.data.frame(hold_out_Mar %>% filter(accommodates > 3) %>% select(price))$price, as.data.frame(hold_out_Mar %>% filter(accommodates > 3) %>% select(price_pred_RF))$price_pred_RF)

# Bagging
RMSE_bag_small <- RMSE(as.data.frame(hold_out_Mar %>% filter(accommodates <= 3) %>% select(price))$price, as.data.frame(hold_out_Mar %>% filter(accommodates <= 3) %>% select(price_pred_bag))$price_pred_bag)

RMSE_bag_large <- RMSE(as.data.frame(hold_out_Mar %>% filter(accommodates > 3) %>% select(price))$price, as.data.frame(hold_out_Mar %>% filter(accommodates > 3) %>% select(price_pred_bag))$price_pred_bag)

## Divide the RMSE by the mean price for small apartments and large apartments respectively.

# OLS
RMSE_overprice_OLS_small <- RMSE_OLS_small/Mean_price_small

RMS_overpriceE_OLS_large <- RMSE_OLS_large/Mean_price_large

# RF
RMSE_overprice_RF_small <- RMSE_RF_small/Mean_price_small

RMSE_overprice_RF_large <- RMSE_RF_large/Mean_price_large

# Bagging
RMSE_overprice_bag_small <- RMSE_bag_small/Mean_price_small

RMSE_overprice_bag_large <- RMSE_bag_large/Mean_price_large

## Compile the results (RMSE, RMSE/Mean(Price)) in a table. The rows are small and large apartments.

## RMSE
# For small apartments
RMSE_Holdout_small <- data.frame(RMSE_OLS_small, RMSE_RF_small, RMSE_bag_small) %>% 
  rename(OLS = RMSE_OLS_small, RF = RMSE_RF_small, Bagging = RMSE_bag_small)


# For large apartments
RMSE_Holdout_large <- data.frame(RMSE_OLS_large, RMSE_RF_large, RMSE_bag_large) %>% 
  rename(OLS = RMSE_OLS_large, RF = RMSE_RF_large, Bagging = RMSE_bag_large)


# Compile the results in a table
RMSE_Holdout_size <- rbind(RMSE_Holdout_small, RMSE_Holdout_large) 

# Renaming rows
rownames(RMSE_Holdout_size) <- c("Small Apartments", "Large Apartments")

## RMSE/Mean(Price)
# For small apartments
RMSE_overprice_Holdout_small <- data.frame(RMSE_overprice_OLS_small, RMSE_overprice_RF_small, RMSE_overprice_bag_small) %>% 
  rename(OLS = Mean_price_small, RF = `Mean_price_small.1`, Bagging = `Mean_price_small.2`)


# For large apartments
RMSE_overprice_Holdout_large <- data.frame(RMS_overpriceE_OLS_large, RMSE_overprice_RF_large, RMSE_overprice_bag_large) %>% 
  rename(OLS = Mean_price_large, RF = `Mean_price_large.1`, Bagging = `Mean_price_large.2`)


# Compile the results in a table
RMSE_overprice_Holdout_size <- rbind(RMSE_overprice_Holdout_small, RMSE_overprice_Holdout_large)

# Renaming rows
rownames(RMSE_overprice_Holdout_size) <- c("Small Apartments", "Large Apartments")


## Compile all results in a table
Holdout_size <- cbind(RMSE_Holdout_size, RMSE_overprice_Holdout_size)

# Renaming columns
colnames(Holdout_size) <- c("RMSE_OLS", "RMSE_RF", "RMSE_Bagging", "RMSE/Price_OLS", "RMSE/Price_RF", "RMSE/Price_Bagging")


## Combine RMSE_Holdout and Holdout_size in a table.
# Transform RMSE_Holdout to wide format.

RMSE_Holdout_wide <- RMSE_Holdout %>% 
  pivot_wider(names_from = model, values_from = c(RMSE, `RMSE/Price`))

# Add the row name
rownames(RMSE_Holdout_wide) <- c("All Apartments")


# Combine the two tables
Holdout_All_results <- rbind(Holdout_size, RMSE_Holdout_wide)

# Add the mean price column
Holdout_All_results <- Holdout_All_results %>% 
  mutate(Mean_price = c(Mean_price_small, Mean_price_large, mean(hold_out_Mar$price)))

# Make the mean price column to be numeric
Holdout_All_results$Mean_price <- as.numeric(Holdout_All_results$Mean_price)

## Print the results 
# Use kable with kableExtra for better formatting
kable(Holdout_All_results, digits = 2, align = "c") %>%
  kable_styling(latex_options = c("scale_down", "hold_position", "striped"))



### Plotting the actual price vs predicted price.

# Have 3 plots. One for each model.

## OLS
hold_out_Mar %>%
  ggplot(aes(x = price, y = price_pred_ols)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "OLS - Actual vs Predicted Prices (March 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, NA) + ylim(0, NA)  

## Plot results from Random Forest 
hold_out_Mar %>% 
  ggplot(aes(x = price, y = price_pred_RF)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "Random Forest - Actual vs Predicted Prices (March 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, NA) + ylim(0, NA)  

## Plot results from Bagging

hold_out_Mar %>% 
  ggplot(aes(x = price, y = price_pred_bag)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "Bagging - Actual vs Predicted Prices (March 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, NA) + ylim(0, NA)  

###### Task 2
### The live data is data from September 2023. 
## Import the data
Data_Sept <- read_csv("listings_sept_2023.csv", show_col_types = FALSE, na = c("", "NA", "N/A"))

## Repeat the same data cleaning process as in Task 1. Keep only apartments.
Data_Sept = Data_Sept %>% 
  filter(property_type %in% c("Entire serviced apartment", "Private room in serviced apartment", "Room in serviced apartment", "Shared room in serviced apartment", "Entire home/apt", "Entire condo", "Private room in condo", "Shared room in condo"), accommodates >= 2, accommodates <= 6)


# Convert the price column to numeric
Data_Sept$price <- gsub("\\$", "", Data_Sept$price)
Data_Sept$price <- as.numeric(gsub(",", "", Data_Sept$price))



# Keep the columns that are used in the model
Data_Sept = Data_Sept %>% 
  select(price, host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost, host_total_listings_count, host_identity_verified, neighbourhood_cleansed, property_type, room_type, accommodates, bathrooms_text, bedrooms, beds, amenities, minimum_nights, maximum_nights, has_availability, availability_30, availability_60, availability_90, availability_365, number_of_reviews, last_review, last_scraped, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, instant_bookable, description, neighborhood_overview)


# Convert the host_response_rate column to numeric
Data_Sept$host_response_rate <- gsub("\\%", "", Data_Sept$host_response_rate)

Data_Sept$host_response_rate <- as.numeric(gsub(",", "", Data_Sept$host_response_rate))

# Convert the host_acceptance_rate column to numeric
Data_Sept$host_acceptance_rate <- gsub("\\%", "", Data_Sept$host_acceptance_rate)

Data_Sept$host_acceptance_rate <- as.numeric(gsub(",", "", Data_Sept$host_acceptance_rate))


# Initialize new columns
Data_Sept$num_baths <- numeric(nrow(Data_Sept))
Data_Sept$num_shared_baths <- numeric(nrow(Data_Sept))
Data_Sept$num_private_baths <- numeric(nrow(Data_Sept))
Data_Sept$is_half_bath <- numeric(nrow(Data_Sept))

# Extract information using regular expressions
for (i in seq_along(Data_Sept$bathrooms_text)) {
  # Check for NA
  if (!is.na(Data_Sept$bathrooms_text[i])) {
    # Extract number and type of bath
    match <- regmatches(Data_Sept$bathrooms_text[i], regexec("(\\d+)\\s*(shared|private)?\\s*baths?", Data_Sept$bathrooms_text[i]))[[1]]
    
    # Extract number
    num <- as.numeric(match[2])
    
    # Determine type and assign to corresponding variable
    if (grepl("shared", match[3], ignore.case = TRUE)) {
      Data_Sept$num_shared_baths[i] <- num
    } else if (grepl("private", match[3], ignore.case = TRUE)) {
      Data_Sept$num_private_baths[i] <- num
    } else {
      Data_Sept$num_baths[i] <- num
    }
    
    # Check if it is a half bath
    Data_Sept$is_half_bath[i] <- ifelse(grepl("half", tolower(Data_Sept$bathrooms_text[i])) || grepl("shared half", tolower(Data_Sept$bathrooms_text[i])), 1, 0)
  }
}

# Set values to NA if bathrooms_text is NA
Data_Sept$num_baths[is.na(Data_Sept$bathrooms_text)] <- NA
Data_Sept$num_shared_baths[is.na(Data_Sept$bathrooms_text)] <- NA
Data_Sept$num_private_baths[is.na(Data_Sept$bathrooms_text)] <- NA
Data_Sept$is_half_bath[is.na(Data_Sept$bathrooms_text)] <- NA

# Drop the original 'bathrooms_text' column
Data_Sept <- subset(Data_Sept, select = -bathrooms_text)

## Extracting amenities into dummy variables
# List of amenities
amenities_list <- c("Hair dryer", "Shampoo", "Shower gel", "Air conditioning", "Essentials", 
                    "Wifi", "Washer", "Iron", "Smoking allowed", "Free parking on premises", 
                    "Luggage dropoff allowed", "Kitchen", "Refrigerator", "Dining table", 
                    "Dedicated workspace", "Elevator", "Microwave", "Dishes and silverware", 
                    "TV", "Pool", "Gym", "Pets allowed")

for (amenity in amenities_list) {
  dummy_variable_name <- paste0(gsub(" ", "_", amenity), "_dummy")
  Data_Sept[[dummy_variable_name]] <- ifelse(sapply(Data_Sept$amenities, grepl, pattern = amenity), 1, 0)
}

# Drop the original 'amenities' column
Data_Sept <- subset(Data_Sept, select = -amenities)

## Encoding other variables into dummy variables
# host_response_time
Data_Sept = dummy_cols(Data_Sept, select_columns = "host_response_time", remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# drop host_response_time_NA
Data_Sept <- subset(Data_Sept, select = -host_response_time_NA)

# neighbourhood_cleansed
Data_Sept = dummy_cols(Data_Sept, select_columns = "neighbourhood_cleansed", remove_first_dummy = TRUE, remove_selected_columns = TRUE)


# property_type
Data_Sept = dummy_cols(Data_Sept, select_columns = "property_type", remove_first_dummy = TRUE)

# room_type
Data_Sept = dummy_cols(Data_Sept, select_columns = "room_type", remove_first_dummy = TRUE, remove_selected_columns = TRUE)


## Creating variable called "time_since_last_review"
# Convert data to date first
Data_Sept$last_review <- as.Date(Data_Sept$last_review, "%Y-%b-%d")

Data_Sept$last_scraped <- as.Date(Data_Sept$last_scraped, "%Y-%b-%d")

# Calculate the time between the two dates
Data_Sept$time_since_last_review <- Data_Sept$last_scraped - Data_Sept$last_review

# Removing the used two columns
Data_Sept = Data_Sept %>%
  select(-c(last_review, last_scraped))


## Extrating Information about Skytrain, Metro
Data_Sept$near_MRT <- ifelse(
  grepl(paste(MRT_keywords, collapse = "|"), 
        Data_Sept$description, ignore.case = TRUE) |
    grepl(paste(MRT_keywords, collapse = "|"), 
          Data_Sept$neighborhood_overview, ignore.case = TRUE),
  1, 0
)

Data_Sept$near_BTS <- ifelse(
  grepl(paste(BTS_keywords, collapse = "|"), 
        Data_Sept$description, ignore.case = TRUE) |
    grepl(paste(BTS_keywords, collapse = "|"), 
          Data_Sept$neighborhood_overview, ignore.case = TRUE),
  1, 0
)

Data_Sept$near_ARL <- ifelse(
  grepl(paste(ARL_keywords, collapse = "|"), 
        Data_Sept$description, ignore.case = TRUE) |
    grepl(paste(ARL_keywords, collapse = "|"), 
          Data_Sept$neighborhood_overview, ignore.case = TRUE),
  1, 0
)


Data_Sept <- Data_Sept %>%
  mutate(near_MRT = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_MRT))

Data_Sept <- Data_Sept %>%
  mutate(near_BTS = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_BTS))

Data_Sept <- Data_Sept %>%
  mutate(near_ARL = ifelse(is.na(description) & is.na(neighborhood_overview), NA, near_ARL))

Data_Sept = Data_Sept %>% select(-c(description, neighborhood_overview))

## Dealing with NAs
# Drop those with missing values for host_acceptance_rate and host_response_rate
Data_Sept <- Data_Sept %>% 
  filter(!is.na(host_acceptance_rate) & !is.na(host_response_rate))

# Drop those with missing values for baths
Data_Sept <- Data_Sept %>% 
  filter(!is.na(num_baths) & !is.na(num_shared_baths) & !is.na(is_half_bath) & !is.na(num_private_baths))

# Replace NA values in near_MRT, near_BTS, and near_ARL with 0

Data_Sept <- Data_Sept %>% 
  mutate(across(c(near_MRT, near_BTS, near_ARL), ~ ifelse(is.na(.), 0, .)))

# Replace NA values in host_is_superhost with 0
Data_Sept <- Data_Sept %>% 
  mutate(host_is_superhost = ifelse(is.na(host_is_superhost), 0, host_is_superhost))

# Replace NA values in time_since_last_review with max
Data_Sept <- Data_Sept %>% 
  mutate(time_since_last_review = ifelse(is.na(time_since_last_review), max(time_since_last_review, na.rm = TRUE), time_since_last_review))

## Replace NA values in scores with mean
Data_Sept <- Data_Sept %>% 
  mutate(across(c(review_scores_rating, review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Create a binary flag variable to capture the imputation
Data_Sept <- Data_Sept %>% 
  mutate(review_scores_rating_imputed = ifelse(is.na(review_scores_rating), 1, 0))

## Replace NA values in bedrooms and beds with mean
Data_Sept <- Data_Sept %>% 
  mutate(across(c(bedrooms, beds), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Create a binary flag variable to capture the imputation
Data_Sept <- Data_Sept %>% 
  mutate(bedrooms_imputed = ifelse(is.na(bedrooms), 1, 0))

### Keeping only apartments
Data_Sept = Data_Sept %>% 
  filter(property_type %in% c("Entire serviced apartment", "Private room in serviced apartment", "Room in serviced apartment", "Shared room in serviced apartment"))

# Drop property_type
Data_Sept <- subset(Data_Sept, select = -property_type)


### The work set and the live data may have different columns. For example, there might not be observations from some neighbourhooods or observations with certain amenities in the live data. Therefore, we need to make sure that the work set and the live data have the same columns. 

## Create a list of columns that are in the work set but not in the live data
cols_to_add_live <- setdiff(colnames(Data_Mar), colnames(Data_Sept))

## Add the columns to the live data
Data_Sept[cols_to_add_live] <- 0




## Predict the price using OLS Model.
Data_Sept$price_pred_ols <- predict(model_OLS_optm, newdata = Data_Sept)

## Predict the price using Random Forest Model.
Data_Sept$price_pred_RF <- predict(model_RF_optm, newdata = Data_Sept)

## Predict the price using Bagging Model.
Data_Sept$price_pred_bag <- predict(model_bagging_optm, newdata = Data_Sept)


### Repeat the same process of calculating and compiling RMSEs.

## Calculate the RMSE for OLS Model.
RMSE_OLS_Live <- RMSE(Data_Sept$price, Data_Sept$price_pred_ols)

# Calculate the RMSE/Mean(Price)
'RMSE_OLS/Mean(Price)' <- RMSE_OLS_Live/mean(Data_Sept$price)

## Calculate the RMSE for Random Forest Model.
RMSE_RF_Live <- RMSE(Data_Sept$price, Data_Sept$price_pred_RF)

# Calculate the RMSE/Mean(Price)
'RMSE_RF/Mean(Price)' <- RMSE_RF_Live/mean(Data_Sept$price)

## Calculate the RMSE for Bagging Model.
RMSE_bag_Live <- RMSE(Data_Sept$price, Data_Sept$price_pred_bag)

# Calculate the RMSE/Mean(Price)
'RMSE_bag/Mean(Price)' <- RMSE_bag_Live/mean(Data_Sept$price)

## Compile the RMSE results and RMSE/Mean(Price) results for the live data.

RMSE_Live <- data.frame(RMSE_OLS_Live, RMSE_RF_Live, RMSE_bag_Live) %>%
  pivot_longer(everything(), names_to = "model", values_to = "RMSE") %>%
  mutate(model = c("OLS", "RF", "Bagging"))


# add column for RMSE/Mean(Price)
RMSE_Live$'RMSE/Price' <- c(`RMSE_OLS/Mean(Price)`, `RMSE_RF/Mean(Price)`, `RMSE_bag/Mean(Price)`)

# Transform RMSE_Live to wide format.

RMSE_Live_wide <- RMSE_Live %>% 
  pivot_wider(names_from = model, values_from = c(RMSE, `RMSE/Price`))


# Add the row name
rownames(RMSE_Live_wide) <- c("All Apartments")

### Compare the performance across apartment size (small apartments: M= 3, large apartments > 3)

## Calculate the Mean Price for small apartments and large apartments in the live data.

Mean_price_small <- Data_Sept %>% 
  filter(accommodates <= 3) %>% 
  summarise(Mean_price_small = mean(price))

Mean_price_large <- Data_Sept %>% 
  filter(accommodates > 3) %>% 
  summarise(Mean_price_large = mean(price))

## For each model, calculate the RMSE for small apartments and large apartments in the live data using the RMSE function.

# OLS

RMSE_OLS_small <- RMSE(as.data.frame(Data_Sept %>% filter(accommodates <= 3) %>% select(price))$price, as.data.frame(Data_Sept %>% filter(accommodates <= 3) %>% select(price_pred_ols))$price_pred_ols)

RMSE_OLS_large <- RMSE(as.data.frame(Data_Sept %>% filter(accommodates > 3) %>% select(price))$price, as.data.frame(Data_Sept %>% filter(accommodates > 3) %>% select(price_pred_ols))$price_pred_ols)


# RF

RMSE_RF_small <- RMSE(as.data.frame(Data_Sept %>% filter(accommodates <= 3) %>% select(price))$price, as.data.frame(Data_Sept %>% filter(accommodates <= 3) %>% select(price_pred_RF))$price_pred_RF)

RMSE_RF_large <- RMSE(as.data.frame(Data_Sept %>% filter(accommodates > 3) %>% select(price))$price, as.data.frame(Data_Sept %>% filter(accommodates > 3) %>% select(price_pred_RF))$price_pred_RF)


# Bagging

RMSE_bag_small <- RMSE(as.data.frame(Data_Sept %>% filter(accommodates <= 3) %>% select(price))$price, as.data.frame(Data_Sept %>% filter(accommodates <= 3) %>% select(price_pred_bag))$price_pred_bag)

RMSE_bag_large <- RMSE(as.data.frame(Data_Sept %>% filter(accommodates > 3) %>% select(price))$price, as.data.frame(Data_Sept %>% filter(accommodates > 3) %>% select(price_pred_bag))$price_pred_bag)

## Divide the RMSE by the mean price for small apartments and large apartments respectively.

# OLS
RMSE_overprice_OLS_small <- RMSE_OLS_small/Mean_price_small

RMS_overpriceE_OLS_large <- RMSE_OLS_large/Mean_price_large

# RF
RMSE_overprice_RF_small <- RMSE_RF_small/Mean_price_small

RMSE_overprice_RF_large <- RMSE_RF_large/Mean_price_large

# Bagging
RMSE_overprice_bag_small <- RMSE_bag_small/Mean_price_small

RMSE_overprice_bag_large <- RMSE_bag_large/Mean_price_large

## Compile the results (RMSE, RMSE/Mean(Price)) in a table. The rows are small and large apartments.

## RMSE
# For small apartments
RMSE_Live_small <- data.frame(RMSE_OLS_small, RMSE_RF_small, RMSE_bag_small) %>% 
  rename(OLS = RMSE_OLS_small, RF = RMSE_RF_small, Bagging = RMSE_bag_small)


# For large apartments
RMSE_Live_large <- data.frame(RMSE_OLS_large, RMSE_RF_large, RMSE_bag_large) %>% 
  rename(OLS = RMSE_OLS_large, RF = RMSE_RF_large, Bagging = RMSE_bag_large)


# Compile the results in a table
RMSE_Live_size <- rbind(RMSE_Live_small, RMSE_Live_large) 

# Renaming rows
rownames(RMSE_Live_size) <- c("Small Apartments", "Large Apartments")

## RMSE/Mean(Price)
# For small apartments
RMSE_overprice_Live_small <- data.frame(RMSE_overprice_OLS_small, RMSE_overprice_RF_small, RMSE_overprice_bag_small) %>% 
  rename(OLS = Mean_price_small, RF = `Mean_price_small.1`, Bagging = `Mean_price_small.2`)


# For large apartments
RMSE_overprice_Live_large <- data.frame(RMS_overpriceE_OLS_large, RMSE_overprice_RF_large, RMSE_overprice_bag_large) %>% 
  rename(OLS = Mean_price_large, RF = `Mean_price_large.1`, Bagging = `Mean_price_large.2`)


# Compile the results in a table
RMSE_overprice_Live_size <- rbind(RMSE_overprice_Live_small, RMSE_overprice_Live_large)

# Renaming rows
rownames(RMSE_overprice_Live_size) <- c("Small Apartments", "Large Apartments")


## Compile all results in a table
Live_size <- cbind(RMSE_Live_size, RMSE_overprice_Live_size)

# Renaming columns
colnames(Live_size) <- c("RMSE_OLS", "RMSE_RF", "RMSE_Bagging", "RMSE/Price_OLS", "RMSE/Price_RF", "RMSE/Price_Bagging")

## Combine RMSE_Live and Live_size in a table.

# Combine the two tables
Live_All_results <- rbind(Live_size, RMSE_Live_wide)

# Add the mean price column
Live_All_results <- Live_All_results %>% 
  mutate(Mean_price = c(Mean_price_small, Mean_price_large, mean(Data_Sept$price)))

# Make the mean price column to be numeric
Live_All_results$Mean_price <- as.numeric(Live_All_results$Mean_price)

## Print the results 
# Use kable with kableExtra for better formatting
kable(Live_All_results, digits = 2, align = "c") %>%
  kable_styling(latex_options = c("scale_down", "hold_position", "striped"))




### Let's plot Yhat vs Y again

## OLS
Data_Sept %>%
  ggplot(aes(x = price, y = price_pred_ols)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "OLS - Actual vs Predicted Prices (September 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, NA) + ylim(0, NA)  

## Plot results from Random Forest 
Data_Sept %>% 
  ggplot(aes(x = price, y = price_pred_RF)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "Random Forest - Actual vs Predicted Prices (September 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, NA) + ylim(0, NA)  

## Plot results from Bagging

Data_Sept %>% 
  ggplot(aes(x = price, y = price_pred_bag)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "Bagging - Actual vs Predicted Prices (September 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, NA) + ylim(0, NA)  


### Plot without outliers
## OLS
Data_Sept %>%
  ggplot(aes(x = price, y = price_pred_ols)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "OLS - Actual vs Predicted Prices (September 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, 20000) + ylim(0, 20000)  



## Plot results from Random Forest 
Data_Sept %>% 
  ggplot(aes(x = price, y = price_pred_RF)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "Random Forest - Actual vs Predicted Prices (September 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, 20000) + ylim(0, 20000)  



## Plot results from Bagging

Data_Sept %>% 
  ggplot(aes(x = price, y = price_pred_bag)) +
  geom_point(color = "steelblue", alpha = 0.7) +  # Adjust point aesthetics
  geom_smooth(method = "lm", se = FALSE, color = "orange", linetype = "dashed", linewidth = 1) +  # Add a smoother line
  labs(title = "Bagging - Actual vs Predicted Prices (September 2023)",
       x = "Actual Price", y = "Predicted Price") +
  theme_minimal() +  # Adjust the theme
  theme(plot.title = element_text(hjust = 0.5),  # Center the title
        axis.text = element_text(size = 10),  # Adjust axis text size
        axis.title = element_text(size = 12, face = "bold")) + xlim(0, 20000) + ylim(0, 20000)  


###### Task 3

## Create a unified model object.
# This is required for the shap functions to work.

unified <- unify(model_RF_optm, data.matrix(hold_out_Mar[,-1]))


## Produce SHAP values
treeshap_all <- treeshap(unified, data.matrix(hold_out_Mar[,-1]))

## Create shapviz object
shp_viz <- shapviz(treeshap_all, X = data.matrix(hold_out_Mar[,-1]))


### Plotting the results.
## Bee swarm plot

sv_importance(shp_viz, kind = "beeswarm")


## Dependence plot b/w time since last review and review scores rating

sv_dependence(shp_viz, "time_since_last_review", color_var = "review_scores_rating")
