## Code for Prediction with Machine Learning _ Assignment 1
# Set the working directory
setwd("D:/Onedrive-CEU/OneDrive - Central European University/CEU/Prediction with Machine Learning/Assignment/DA3-phdma/Assignment 1")

library(readr)
library(dplyr)

# Importing the data

DataOriginal <- read_csv("morg-2014-emp.csv")

# Filter - only select 'Miscellaneous agricultural workers, including animal breeders' - 6050

MiscWorkerData = DataOriginal %>%
  filter(occ2012 == 6050)
