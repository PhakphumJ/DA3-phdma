###### Task 1
### Importing Data

rm(list=ls())
# Import Libraries
library(readr)
library(dplyr)
library(ggplot2)
library(weights)
library(caret)
library(kableExtra)
library(tidyr)
library(fastDummies)
library(groupdata2)
library(glmnet)
library(pROC)
library(randomForest)

# Set Working Directory

setwd("D:/Onedrive-CEU/OneDrive - Central European University/CEU/Prediction with Machine Learning/Assignment/DA3-phdma/Assignment 3")

# Import Data

Data_panel <- read_csv("cs_bisnode_panel.csv")

## inspect year in Data_panel
Data_panel %>% 
  group_by(year) %>% 
  summarise(n = n())

## add all missing year and comp_id combinations -
Data_panel <- Data_panel %>%
  complete(year, comp_id)

#### Sample Design (part 1)
## Restrict to those with sales > 6,000 eur
Data_panel <- Data_panel %>% 
  filter(sales > 6000)

#### Dealing with Missing Values (Part 1)
### Inspect NAs in Data_panel showing the most NAs first.
Data_panel %>% ungroup() %>%
  summarise_all(funs(sum(is.na(.)))) %>% 
  gather(variable, value) %>% 
  arrange(desc(value))

# the missing values in founded_year, ceo_count, foreign, female, inoffice_days, 
# gender, and origin can imputed by looking at the next year. For example:

## Print the value of founded_year, ceo_count, foreign, female, inoffice_days, gender, and origin of firm 1003200.
Data_panel %>% filter(comp_id == 1003200) %>%
  select(comp_id, year, founded_year, ceo_count, foreign, female, inoffice_days, gender, origin)  


Data_panel %>% filter(comp_id == 1046213) %>%
  select(comp_id, year, founded_year, ceo_count, foreign, female, inoffice_days, gender, origin)

Data_panel %>% filter(comp_id == 1084759
) %>%
  select(comp_id, year, founded_year, ceo_count, foreign, female, inoffice_days, gender, origin)

# Similar things can be observed with *birth_year*. 
# It can be imputed by using the value in the next year. For example:
Data_panel %>% filter(comp_id == 1046213) %>%
  select(comp_id, year, ceo_count, female, birth_year)

### drop variables with too many NAs
Data_panel <- Data_panel %>%
  select(-c(D, COGS, finished_prod, net_dom_sales, net_exp_sales, wages))

### drop irrelevant variables
Data_panel <- Data_panel %>%
  select(-c(exit_year, exit_date))

### Imputing missing values in founded_year, ceo_count, foreign, female, inoffice_days, gender, and origin.

## Impute missing values in founded_year with the lead year.
# repeat 10 times to make sure that all missing values are imputed
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(founded_year = ifelse(is.na(founded_year), lead(founded_year), founded_year),
           flag_miss_founded_year = as.numeric(is.na(founded_year)))
}

## Do the same thing for ceo_count.
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(ceo_count = ifelse(is.na(ceo_count), lead(ceo_count), ceo_count))
}

## Do the same thing for foreign.
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(foreign = ifelse(is.na(foreign), lead(foreign), foreign))
}

## Do the same thing for female.
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(female = ifelse(is.na(female), lead(female), female))
}

## Do the same thing for inoffice_days.
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(inoffice_days = ifelse(is.na(inoffice_days), lead(inoffice_days), inoffice_days))
}

## Do the same thing for gender.
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(gender = ifelse(is.na(gender), lead(gender), gender))
}


## Do the same thing for origin.
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(origin = ifelse(is.na(origin), lead(origin), origin))
}

### Imputing missing values in birth_year.
for (i in 1:10) {
  Data_panel <- Data_panel %>%
    mutate(birth_year = ifelse(is.na(birth_year), lead(birth_year), birth_year),
           flag_miss_birth_year = as.numeric(is.na(birth_year)))
}

### Fixing items that cannot be negative.
## assets can't be negative. Change them to 0 and add a flag.
Data_panel <- Data_panel  %>%
  mutate(flag_asset_problem=ifelse(intang_assets<0 | curr_assets<0 | fixed_assets<0,1,0  ))
table(Data_panel$flag_asset_problem)

Data_panel <- Data_panel %>%
  mutate(intang_assets = ifelse(intang_assets < 0, 0, intang_assets),
         curr_assets = ifelse(curr_assets < 0, 0, curr_assets),
         fixed_assets = ifelse(fixed_assets < 0, 0, fixed_assets))


## Do the same for extra_inc
Data_panel <- Data_panel  %>%
  mutate(flag_extra_inc_problem=ifelse(extra_inc<0,1,0  ))

Data_panel <- Data_panel %>%
  mutate(extra_inc = ifelse(extra_inc < 0, 0, extra_inc))

## Do the same for inventories
Data_panel <- Data_panel  %>%
  mutate(flag_inventories_problem=ifelse(inventories<0,1,0  ))

Data_panel <- Data_panel %>%
  mutate(inventories = ifelse(inventories < 0, 0, inventories))

## Do the same for personnel_exp
Data_panel <- Data_panel  %>%
  mutate(flag_personnel_exp_problem=ifelse(personnel_exp<0,1,0  ))

Data_panel <- Data_panel %>%
  mutate(personnel_exp = ifelse(personnel_exp < 0, 0, personnel_exp))

## Do the same for curr_liab
Data_panel <- Data_panel  %>%
  mutate(flag_curr_liab_problem=ifelse(curr_liab<0,1,0  ))

Data_panel <- Data_panel %>%
  mutate(curr_liab = ifelse(curr_liab < 0, 0, curr_liab))

#### Label Engineering
## Create variable total_cost
Data_panel <- Data_panel %>% 
  mutate(total_cost = sales + extra_inc - profit_loss_year )

## Create variable sales_cost_ratio
Data_panel <- Data_panel %>% 
  mutate(sales_cost_ratio = sales/total_cost)

## Create variable growth_rate in sales_cost_ratio for each firm in each year

## arrange Data_panel by comp_id and year.
Data_panel <- Data_panel %>% 
  arrange(comp_id, year)

## Create the growth rate variable
Data_panel <- Data_panel %>% 
  group_by(comp_id) %>% 
  mutate(growth_rate = (sales_cost_ratio/lag(sales_cost_ratio) - 1)*100)

## Let's plot distribution of growth_rate
Data_panel %>% 
  ggplot(aes(x = growth_rate)) +
  geom_histogram(bins = 50) +
  labs(title = "Distribution of growth_rate",
       x = "growth_rate (%)",
       y = "Count") + xlim(-150, 300)

## Create variable forth_quartile_growth_rate for each industry
Data_panel <- Data_panel %>% 
  group_by(ind, year) %>% 
  mutate(forth_quartile_growth_rate = quantile(growth_rate, 0.75, na.rm = TRUE))


### Create variable fast_growing
## Create variable growth_rate_t+1
Data_panel <- Data_panel %>% 
  group_by(comp_id) %>% 
  mutate(growth_rate_t1 = lead(growth_rate, 1))

## Create variable growth_rate_t+2
Data_panel <- Data_panel %>% 
  group_by(comp_id) %>% 
  mutate(growth_rate_t2 = lead(growth_rate, 2))

## Create variable forth_quartile_growth_rate_t+1
Data_panel <- Data_panel %>% 
  group_by(ind, year) %>% 
  mutate(forth_quartile_growth_rate_t1 = lead(forth_quartile_growth_rate, 1))

## Create variable forth_quartile_growth_rate_t+2
Data_panel <- Data_panel %>% 
  group_by(ind, year) %>% 
  mutate(forth_quartile_growth_rate_t2 = lead(forth_quartile_growth_rate, 2))

## Create variable outstanding_growth_rate_t+1 (growth_rate t+1 > or = forth_quartile_growth_rate_t+1)
Data_panel <- Data_panel %>% 
  mutate(outstanding_growth_rate_t1 = growth_rate_t1 > forth_quartile_growth_rate_t1 | growth_rate_t1 == forth_quartile_growth_rate_t1)


## Create variable outstanding_growth_rate_t+2
Data_panel <- Data_panel %>% 
  mutate(outstanding_growth_rate_t2 = growth_rate_t2 > forth_quartile_growth_rate_t2 | growth_rate_t2 == forth_quartile_growth_rate_t2)


## Create variable fast_growing. Fast growing = 1 if and only if outstanding_growth_rate_t+1 = 1 and outstanding_growth_rate_t+2 = 1, otherwise 0
Data_panel$fast_growing <- ifelse(
  Data_panel$outstanding_growth_rate_t1 == 1 & Data_panel$outstanding_growth_rate_t2 == 1,
  1,
  0
)

# Those with missing values in fast_growing are those who did not appear in t+1 or t+2 because they exit the market. Hence, it should be 0.
Data_panel$fast_growing[is.na(Data_panel$fast_growing)] <- 0

# Do the same for outstanding_growth_rate_t1 and outstanding_growth_rate_t2
Data_panel$outstanding_growth_rate_t1[is.na(Data_panel$outstanding_growth_rate_t1)] <- 0
Data_panel$outstanding_growth_rate_t2[is.na(Data_panel$outstanding_growth_rate_t2)] <- 0

# Drop growth_rate_t1, growth_rate_t2, forth_quartile_growth_rate_t1, forth_quartile_growth_rate_t2
Data_panel <- Data_panel %>% 
  select(-c(growth_rate_t1, growth_rate_t2, forth_quartile_growth_rate_t1, forth_quartile_growth_rate_t2))

## Let's examine outstanding_growth_rate_t1
Data_panel %>% 
  group_by(year) %>%
  count(outstanding_growth_rate_t1)

## Lets' examine fast_growing
Data_panel %>% 
  group_by(year) %>%
  count(fast_growing)

#### Feature Engineering 
Data_panel <- Data_panel %>%
  ungroup()

## Create variable age
Data_panel <- Data_panel %>% 
  mutate(age = year - founded_year)

### CEO age
## Drop those without birth_year
Data_panel <- Data_panel %>%
  filter(!is.na(birth_year))

## Create variable ceo_age
Data_panel <- Data_panel %>%
  mutate(ceo_age = year - birth_year)

## Catagorize ceo_age into 3 intervals. Young: < 35, Middle: 36-65, Old: > 65. Use Old as reference.
Data_panel <- Data_panel %>%
  mutate(ceo_age_young = as.numeric(ceo_age < 35),
         ceo_age_middle = as.numeric(ceo_age >= 35 & ceo_age <= 65),
         ceo_age_old = as.numeric(ceo_age > 65))

## Drop ceo_age
Data_panel <- Data_panel %>%
  select(-ceo_age)

## Drop ceo_age_old
Data_panel <- Data_panel %>%
  select(-ceo_age_old)

## change some industry category codes
Data_panel <- Data_panel %>%
  mutate(ind2_cat = ind2 %>%
           ifelse(. > 56, 60, .)  %>%
           ifelse(. < 26, 20, .) %>%
           ifelse(. < 55 & . > 35, 40, .) %>%
           ifelse(. == 31, 30, .) %>%
           ifelse(is.na(.), 99, .)
  )

# drop ind2
Data_panel <- Data_panel %>%
  select(-ind2)

# Examine value of ind2_cat
Data_panel %>%
  group_by(ind2_cat) %>%
  count()

# Drop those with ind2_cat = 35
Data_panel <- Data_panel %>%
  filter(ind2_cat != 35)

## Make ind2_cat a factor variable
Data_panel <- Data_panel %>%
  mutate(ind2_cat = factor(ind2_cat, levels = c(20, 26, 27, 28, 30, 32, 33, 40, 55, 56, 60, 99)))

### create factors variables urban_m, gender_m, region_m.
Data_panel <- Data_panel %>%
  mutate(gender_m = factor(gender, levels = c("female", "male", "mix")),
         region_m = factor(region_m, levels = c("Central", "East", "West")), urban_m = factor(urban_m, levels = c(1,2,3)), origin_m = factor(origin, levels = c("Foreign", "Domestic", "mix")))

# drop gender
Data_panel <- Data_panel %>% 
  select(-gender)

# drop origin
Data_panel <- Data_panel %>% 
  select(-origin)

### create foreign_management variable (binary)
Data_panel <- Data_panel %>%
  mutate(foreign_management = as.numeric(foreign >= 0.5))

# drop foreign
Data_panel <- Data_panel %>% 
  select(-foreign)

### Create Total Assets

# generate total assets
Data_panel <- Data_panel %>%
  mutate(total_assets = intang_assets + curr_assets + fixed_assets)

### Create Financial Ratios 
## Profitability Ratio: Net Profit Margin.
Data_panel <- Data_panel %>%
  mutate(net_profit_margin = profit_loss_year / sales)

## Current Ratio: Current Assets-to-Current Liabilities Ratio. How well they manage their short-term liquidity.
Data_panel <- Data_panel %>%
  mutate(current_ratio = curr_assets / curr_liab)

# Divide it into 5 categories. 0-1.1, 1.1-2.0, 2.0-4.0, 4.0-6.0, > 6.0 (factor)
Data_panel <- Data_panel %>%
  mutate(current_ratio_cat = case_when(current_ratio <= 1.1 ~ 1,
                                       current_ratio > 1.1 & current_ratio <= 2.0 ~ 2,
                                       current_ratio > 2.0 & current_ratio <= 4.0 ~ 3,
                                       current_ratio > 4.0 & current_ratio <= 6.0 ~ 4,
                                       current_ratio > 6.0 ~ 5))

# make it a factor.
Data_panel <- Data_panel %>%
  mutate(current_ratio_cat = factor(current_ratio_cat, levels = c(1,2,3,4,5) , labels = c("< 1.1", "1.1-2.0", "2.0-4.0", "4.0-6.0", "> 6.0")))

# drop current_ratio
Data_panel <- Data_panel %>%
  select(-current_ratio)

## Solvency Ratio: Equity-to-Assets Ratio.
## Create the variable
Data_panel <- Data_panel %>%
  mutate(equity_to_assets = share_eq / total_assets)

## Divide it into 6 categories. < 0.0, 0.0-0.1, 0.1-0.2, 0.2-0.4, 0.4-0.6, > 0.6 (factor)
Data_panel <- Data_panel %>%
  mutate(equity_to_assets_cat = case_when(equity_to_assets <= 0.0 ~ 1,
                                          equity_to_assets > 0.0 & equity_to_assets <= 0.1 ~ 2,
                                          equity_to_assets > 0.1 & equity_to_assets <= 0.2 ~ 3,
                                          equity_to_assets > 0.2 & equity_to_assets <= 0.4 ~ 4,
                                          equity_to_assets > 0.4 & equity_to_assets <= 0.6 ~ 5,
                                          equity_to_assets > 0.6 ~ 6))

# make it a factor.
Data_panel <- Data_panel %>%
  mutate(equity_to_assets_cat = factor(equity_to_assets_cat, levels = c(1,2,3,4,5,6), labels = c("< 0.0", "0.0-0.1", "0.1-0.2", "0.2-0.4", "0.4-0.6", "> 0.6")))

# drop equity_to_assets
Data_panel <- Data_panel %>%
  select(-equity_to_assets)

## Efficiency Ratio: Inventory-to-Sales Ratio. How well they manage their inventory.
Data_panel <- Data_panel %>%
  mutate(inventory_to_sales = inventories / sales)

## Create Fixed Assets Ratio
Data_panel <- Data_panel %>%
  mutate(fixed_assets_ratio = fixed_assets / total_assets)

## Create Fixed Assets Growth Rate
Data_panel <- Data_panel %>% group_by(comp_id) %>%
  mutate(fixed_assets_growth_rate = (fixed_assets_ratio/lag(fixed_assets_ratio) - 1)*100)

## Create Intangible Assets Ratio
Data_panel <- Data_panel %>%
  mutate(intang_assets_ratio = intang_assets / total_assets)

## Create Intangible Assets Growth Rate
Data_panel <- Data_panel %>% group_by(comp_id) %>%
  mutate(intang_assets_growth_rate = (intang_assets_ratio/lag(intang_assets_ratio) - 1)*100)

### number emp, imputing
## Impute missing values with mean in same industry and similar sales number. And add a flag variable.

# Seperate sales into 5 groups.
Data_panel <- Data_panel %>%
  mutate(sales_group = ntile(sales, 5))

# Impute missing values with mean in same industry, year, and similar sales number. And add a flag variable.
Data_panel <- Data_panel %>%
  group_by(ind, year, sales_group) %>%
  mutate(emp_avg = ifelse(is.na(labor_avg), mean(labor_avg, na.rm = TRUE), labor_avg),
         flag_miss_emp = as.numeric(is.na(labor_avg))) 

Data_panel <- Data_panel %>%
  ungroup()

# drop -labor_avg
Data_panel <- Data_panel %>%
  select(-labor_avg)

# drop sales_group
Data_panel <- Data_panel %>%
  select(-sales_group)

### personnel expenditure per employee
Data_panel <- Data_panel %>%
  mutate(personnel_exp_per_emp = personnel_exp / emp_avg)

### personnel expenditure per employee growth rate
Data_panel <- Data_panel %>% group_by(comp_id) %>%
  mutate(personnel_exp_per_emp_growth_rate = (personnel_exp_per_emp/lag(personnel_exp_per_emp) - 1)*100)

### Growth in number of employees
Data_panel <- Data_panel %>% group_by(comp_id) %>%
  mutate(emp_growth_rate = (emp_avg/lag(emp_avg) - 1)*100)

## Ungroup
Data_panel <- Data_panel %>%
  ungroup()

### There's Inf produced. Replace all Inf in all variables with NA.
Data_panel <- Data_panel %>%
  mutate_all(funs(replace(., is.infinite(.), NA)))

### Dealing with missing values again (Part 2).

## Keep only relevant variables for convenience.
Data_panel_focus <- Data_panel %>% select(comp_id, year, fast_growing, ceo_count, female, inoffice_days, ind2_cat, ind, urban_m, region_m, flag_miss_founded_year, flag_miss_birth_year, growth_rate, age, ceo_age_young, ceo_age_middle, gender_m, origin_m, foreign_management, flag_asset_problem, net_profit_margin, inventory_to_sales, fixed_assets_ratio, fixed_assets_growth_rate, intang_assets_ratio, intang_assets_growth_rate, emp_avg, emp_growth_rate, flag_miss_emp, personnel_exp_per_emp, personnel_exp_per_emp_growth_rate, current_ratio_cat, equity_to_assets_cat, flag_personnel_exp_problem, flag_curr_liab_problem, flag_inventories_problem, flag_extra_inc_problem)

# remove Data_panel
rm(Data_panel)

## Inspect NAs in Data_panel showing the most NAs first. Do it again.
Data_panel_focus %>% ungroup() %>%
  summarise_all(funs(sum(is.na(.)))) %>% 
  gather(variable, value) %>% 
  arrange(desc(value))

## This is weird. I think this is because many firms don't have intangible assets, fixed assets, or employees (denominator = 0). Let's check this. 

# count number of firms with intangible assets ratio = 0.
Data_panel_focus %>% ungroup() %>%
  filter(intang_assets_ratio == 0) %>%
  summarise(n = n())


# My guess is correct. Hence intang_assets_growth_rate should be 0 for these firms. Note that growth = 0 for these firms is not the same as growth = 0 for firm with already high intangible assets ratio. (We will interact this)

# Replace intang_assets_growth_rate with 0 for firm with lag intang_assets_ratio = 0 and intang_assets_ratio = 0. 

Data_panel_focus <- Data_panel_focus %>% group_by(comp_id) %>%
  mutate(intang_assets_growth_rate = ifelse(lag(intang_assets_ratio) == 0 & intang_assets_ratio == 0, 0, intang_assets_growth_rate))

## Let's examine fixed_assets_ratio, count number of firms with fixed_assets_ratio = 0.
Data_panel_focus %>% ungroup() %>%
  filter(fixed_assets_ratio == 0) %>%
  summarise(n = n())

# Replace fixed_assets_growth_rate with 0 for firm with lag fixed_assets_ratio = 0 and fixed_assets_ratio = 0.

Data_panel_focus <- Data_panel_focus %>% group_by(comp_id) %>%
  mutate(fixed_assets_growth_rate = ifelse(lag(fixed_assets_ratio) == 0 & fixed_assets_ratio == 0, 0, fixed_assets_growth_rate))

## Let's examine personnel_exp_per_emp, count number of firms with personnel_exp_per_emp = 0.
Data_panel_focus %>% ungroup() %>%
  filter(personnel_exp_per_emp == 0) %>%
  summarise(n = n())

# Replace personnel_exp_per_emp_growth_rate with 0 for firm with lag personnel_exp_per_emp = 0 and personnel_exp_per_emp = 0.

Data_panel_focus <- Data_panel_focus %>% group_by(comp_id) %>%
  mutate(personnel_exp_per_emp_growth_rate = ifelse(lag(personnel_exp_per_emp) == 0 & personnel_exp_per_emp == 0, 0, personnel_exp_per_emp_growth_rate))

# count number of firms with employees = 0.
Data_panel_focus %>% ungroup() %>%
  filter(emp_avg == 0) %>%
  summarise(n = n())

## Restrict Data_panel_focus to 2010-2013

Data_panel_focus <- Data_panel_focus %>% 
  filter(year >= 2010 & year <= 2013)

## Drop those with missing values in growth_rate since these are firms who did not appear in year t-1. 
Data_panel_focus <- Data_panel_focus %>% 
  filter(!is.na(growth_rate))

## Inspect NAs in Data_panel showing the most NAs first. Do it again.
Data_panel_focus %>% ungroup() %>%
  summarise_all(funs(sum(is.na(.)))) %>% 
  gather(variable, value) %>% 
  arrange(desc(value))

## Drop observations with missing values.
Data_panel_focus <- Data_panel_focus %>% 
  drop_na()

### Winsorize and Simplyfying Variables 
### Let's examine the distribution of the variables first.
## Let's plot inventory_to_sales
Data_panel_focus %>%
  ggplot(aes(x = inventory_to_sales)) +
  geom_histogram(binwidth = 0.05) +
  labs(title = "Distribution of Inventory-to-Sales Ratio",
       x = "Inventory-to-Sales Ratio",
       y = "Count") + xlim(0, 1.5) + ylim(0, 7500)
# There's a few outliers. I winsorize at 1.00.

## Winsorize inventory_to_sales at 1.00. Create a flag variable.

Data_panel_focus <- Data_panel_focus %>% 
  mutate(inventory_to_sales_flag = ifelse(inventory_to_sales > 1.00, 1, 0),
         inventory_to_sales = ifelse(inventory_to_sales > 1.00, 1.00, inventory_to_sales))

## Let's plot net_profit_margin
Data_panel_focus %>%
  ggplot(aes(x = net_profit_margin)) +
  geom_histogram(binwidth = 0.1) +
  labs(title = "Distribution of Net Profit Margin",
       x = "Net Profit Margin",
       y = "Count") + xlim(-3, 2) + ylim(0, 10000)

# Winsorize at -1.50 and 0.80.

## Winsorize net_profit_margin at -1.50 and 0.80. Create two flag variables.

Data_panel_focus <- Data_panel_focus %>% 
  mutate(net_profit_margin_flag_low = ifelse(net_profit_margin < -1.50, 1, 0),
         net_profit_margin_flag_high = ifelse(net_profit_margin > 0.80, 1, 0),
         net_profit_margin = ifelse(net_profit_margin < -1.50, -1.50, net_profit_margin),
         net_profit_margin = ifelse(net_profit_margin > 0.80, 0.80, net_profit_margin))

## Let's plot fixed_assets_ratio
Data_panel_focus %>%
  ggplot(aes(x = fixed_assets_ratio)) +
  geom_histogram(binwidth = 0.05) +
  labs(title = "Distribution of Fixed Assets Ratio",
       x = "Fixed Assets Ratio",
       y = "Count") + xlim(0, 1.5) + ylim(0, 7500)
# Leave it as it is.

## Let's plot fixed_assets_growth_rate
Data_panel_focus %>%
  ggplot(aes(x = fixed_assets_growth_rate)) +
  geom_histogram(binwidth = 5) +
  labs(title = "Distribution of Fixed Assets Growth Rate",
       x = "Fixed Assets Growth Rate",
       y = "Count") + xlim(-150, 150) + ylim(0, 5000)

# Catagorize into 5 groups: -100%, -99% to -50%, -50% to 0%, 0% to 50%, > 50%.

## Catagorize fixed_assets_growth_rate into 5 groups: -100%, -99% to -50%, -50% to 0%, 0% to 50%, > 50%.

Data_panel_focus <- Data_panel_focus %>%
  mutate(fixed_assets_growth_rate_cat = case_when(fixed_assets_growth_rate == -100 ~ 1,
                                                  fixed_assets_growth_rate > -100 & fixed_assets_growth_rate <= -50 ~ 2,
                                                  fixed_assets_growth_rate > -50 & fixed_assets_growth_rate <= 0 ~ 3,
                                                  fixed_assets_growth_rate > 0 & fixed_assets_growth_rate <= 50 ~ 4,
                                                  fixed_assets_growth_rate > 50 ~ 5))

# make it a factor.
Data_panel_focus <- Data_panel_focus %>%
  mutate(fixed_assets_growth_rate_cat = factor(fixed_assets_growth_rate_cat, levels = c(1,2,3,4,5), labels = c("-100%", "-99% to -50%", "-50% to 0%", "0% to 50%", "> 50%")))

# drop fixed_assets_growth_rate
Data_panel_focus <- Data_panel_focus %>%
  select(-fixed_assets_growth_rate)

## Let's plot intang_assets_ratio
Data_panel_focus %>%
  ggplot(aes(x = intang_assets_ratio)) +
  geom_histogram(binwidth = 0.05) +
  labs(title = "Distribution of Intangible Assets Ratio",
       x = "Intangible Assets Ratio",
       y = "Count") + xlim(0, 1) + ylim(0, 3000)

# Leave it as it is.

## Let's plot intang_assets_growth_rate
Data_panel_focus %>%
  ggplot(aes(x = intang_assets_growth_rate)) +
  geom_histogram(binwidth = 5) +
  labs(title = "Distribution of Intangible Assets Growth Rate",
       x = "Intangible Assets Growth Rate",
       y = "Count") + xlim(-110, 150) + ylim(0, 5000)

# Catagorize into 5 groups: -100%, -99% to -50%, -50% to 0%, 0% to 50%, > 50%.

## Catagorize intang_assets_growth_rate into 5 groups: -100%, -99% to -50%, -50% to 0%, 0% to 50%, > 50%.

Data_panel_focus <- Data_panel_focus %>%
  mutate(intang_assets_growth_rate_cat = case_when(intang_assets_growth_rate == -100 ~ 1,
                                                   intang_assets_growth_rate > -100 & intang_assets_growth_rate <= -50 ~ 2,
                                                   intang_assets_growth_rate > -50 & intang_assets_growth_rate <= 0 ~ 3,
                                                   intang_assets_growth_rate > 0 & intang_assets_growth_rate <= 50 ~ 4,
                                                   intang_assets_growth_rate > 50 ~ 5))

# make it a factor.
Data_panel_focus <- Data_panel_focus %>%
  mutate(intang_assets_growth_rate_cat = factor(intang_assets_growth_rate_cat, levels = c(1,2,3,4,5), labels = c("-100%", "-99% to -50%", "-50% to 0%", "0% to 50%", "> 50%")))

# drop intang_assets_growth_rate
Data_panel_focus <- Data_panel_focus %>%
  select(-intang_assets_growth_rate)

## Let's plot emp_growth_rate
Data_panel_focus %>%
  ggplot(aes(x = emp_growth_rate)) +
  geom_histogram(binwidth = 5) +
  labs(title = "Distribution of Employee Growth Rate",
       x = "Employee Growth Rate",
       y = "Count") + xlim(-150, 150) + ylim(0, 5000)

# Leave it as it is.

## Let's plot personnel_exp_per_emp
Data_panel_focus %>%
  ggplot(aes(x = personnel_exp_per_emp)) +
  geom_histogram(binwidth = 5000) +
  labs(title = "Distribution of Personnel Expense per Employee",
       x = "Personnel Expense per Employee",
       y = "Count") + xlim(0,200000) + ylim(0, 7500)

# Leave it as it is.

## Let's plot personnel_exp_per_emp_growth_rate
Data_panel_focus %>%
  ggplot(aes(x = personnel_exp_per_emp_growth_rate)) +
  geom_histogram(binwidth = 5) +
  labs(title = "Distribution of Personnel Expense per Employee Growth Rate",
       x = "Personnel Expense per Employee Growth Rate",
       y = "Count") + xlim(-150, 150) + ylim(0, 5000)

# Catagorize into 5 groups: -100%, -99% to -50%, -50% to 0%, 0% to 50%, > 50%.
## Catagorize personnel_exp_per_emp_growth_rate into 5 groups: -100%, -99% to -50%, -50% to 0%, 0% to 50%, > 50%.

Data_panel_focus <- Data_panel_focus %>%
  mutate(personnel_exp_per_emp_growth_rate_cat = case_when(personnel_exp_per_emp_growth_rate == -100 ~ 1,
                                                           personnel_exp_per_emp_growth_rate > -100 & personnel_exp_per_emp_growth_rate <= -50 ~ 2,
                                                           personnel_exp_per_emp_growth_rate > -50 & personnel_exp_per_emp_growth_rate <= 0 ~ 3,
                                                           personnel_exp_per_emp_growth_rate > 0 & personnel_exp_per_emp_growth_rate <= 50 ~ 4,
                                                           personnel_exp_per_emp_growth_rate > 50 ~ 5))

# make it a factor.
Data_panel_focus <- Data_panel_focus %>%
  mutate(personnel_exp_per_emp_growth_rate_cat = factor(personnel_exp_per_emp_growth_rate_cat, levels = c(1,2,3,4,5), labels = c("-100%", "-99% to -50%", "-50% to 0%", "0% to 50%", "> 50%")))

# drop personnel_exp_per_emp_growth_rate
Data_panel_focus <- Data_panel_focus %>%
  select(-personnel_exp_per_emp_growth_rate)

#### Sample Design (part 2) & Downsampling
## Lets' examine fast_growing again
Data_panel_focus %>% 
  group_by(year) %>%
  count(fast_growing)

## Use only data from 2012.
Data_2012 <- Data_panel_focus %>%
  filter(year == 2012)

# remove Data_panel_focus from the environment
rm(Data_panel_focus)

### Restricting sample and downsampling.
## Downsample majority class in each year
# Randomly filter out 55% of the majority class in each year.
set.seed(20231210)
Data_toss <- Data_2012 %>%
  group_by(year) %>%
  filter(fast_growing == FALSE) %>%
  sample_frac(0.55)

# Toss them out of the original dataset.
Data_2012 <- Data_2012 %>%
  anti_join(Data_toss, by = c("comp_id", "year"))

# remove Data_toss from the environment
rm(Data_toss)

Data_2012 <- Data_2012 %>% ungroup()

## Lets' examine fast_growing again
Data_2012 %>% 
  count(fast_growing)

# dropping flags with no variation
Data_2012 <- Data_2012 %>% ungroup()

variances<- Data_2012 %>%
  select(contains("flag")) %>%
  apply(2, var, na.rm = TRUE) == 0

Data_2012 <- Data_2012 %>%
  select(-one_of(names(variances)[variances]))

#### Building Models
## Split the data into a work set and a hold-out set.
set.seed(20231210)

Data_2012_hold_out <- Data_2012 %>%
  sample_frac(0.15)

# Subtract the hold-out set from the work set (anti_join).
Data_2012 <- Data_2012 %>%
  anti_join(Data_2012_hold_out, by = c("comp_id", "year"))

# drop comp_id and year.

Data_2012 <- Data_2012 %>%
  select(-comp_id, -year)

Data_2012_hold_out <- Data_2012_hold_out %>%
  select(-comp_id, -year)

#### Logistic Regression
### Perform 5-fold cross validation on logistic regression models.

set.seed(20231210)
# Getting the indices for 5-fold cross validation
folds <- createFolds(Data_2012$fast_growing, k = 5, list = TRUE, returnTrain = TRUE)

### Estimating Models and obtains RMSE (5-folds CV)
## For each of these calculate the RMSE and AUC for each fold and average them at the end.
## Model 1

results <- list()
results_auc <- list()

for (i in 1:5) {
  # get the training set
  train <- Data_2012[folds[[i]],]
  
  # get the test set
  test <- Data_2012[-folds[[i]],]
  
  # build the model
  logit_1 <- glm(fast_growing ~ growth_rate, data = train, family = "binomial")
  
  # predict the test set
  pred <- predict(logit_1, test, type = "response")
  
  # calculate RMSE
  rmse <- sqrt(mean((pred - test$fast_growing)^2))
  
  # calculate AUC
  auc <- auc(test$fast_growing, pred)
  
  # store the results
  results[[i]] <- rmse
  
  results_auc[[i]] <- auc
}

# calculate the average RMSE
RMSE_Logit_1 <- sqrt(mean(unlist(results)^2))

# calculate the average AUC
AUC_Logit_1 <- mean(unlist(results_auc))


## Model 2

results <- list()
results_auc <- list()

for (i in 1:5) {
  # get the training set
  train <- Data_2012[folds[[i]],]
  
  # get the test set
  test <- Data_2012[-folds[[i]],]
  
  # build the model
  logit_2 <- update(logit_1, . ~ . + net_profit_margin + current_ratio_cat + equity_to_assets_cat + flag_asset_problem + flag_curr_liab_problem + net_profit_margin_flag_low + net_profit_margin_flag_high, data = train)
  
  # predict the test set
  pred <- predict(logit_2, test, type = "response")
  
  # calculate RMSE
  rmse <- sqrt(mean((pred - test$fast_growing)^2))
  
  # calculate AUC
  auc <- auc(test$fast_growing, pred)
  
  # store the results
  results[[i]] <- rmse
  
  results_auc[[i]] <- auc
}

# calculate the average RMSE
RMSE_Logit_2 <- sqrt(mean(unlist(results)^2))

# calculate the average AUC
AUC_Logit_2 <- mean(unlist(results_auc))

## Model 3 

results <- list()
results_auc <- list()

for (i in 1:5) {
  # get the training set
  train <- Data_2012[folds[[i]],]
  
  # get the test set
  test <- Data_2012[-folds[[i]],]
  
  # build the model
  logit_3 <- update(logit_2, . ~ . + fixed_assets_ratio + intang_assets_ratio + emp_avg + emp_growth_rate + flag_miss_emp + personnel_exp_per_emp + flag_personnel_exp_problem + fixed_assets_growth_rate_cat + intang_assets_growth_rate_cat + personnel_exp_per_emp_growth_rate_cat, data = train)
  
  # predict the test set
  pred <- predict(logit_3, test, type = "response")
  
  # calculate RMSE
  rmse <- sqrt(mean((pred - test$fast_growing)^2))
  
  # calculate AUC
  auc <- auc(test$fast_growing, pred)
  
  # store the results
  results[[i]] <- rmse
  
  results_auc[[i]] <- auc
}

# calculate the average RMSE
RMSE_Logit_3 <- sqrt(mean(unlist(results)^2))

# calculate the average AUC
AUC_Logit_3 <- mean(unlist(results_auc))

## Model 4

results <- list()
results_auc <- list()

for (i in 1:5) {
  # get the training set
  train <- Data_2012[folds[[i]],]
  
  # get the test set
  test <- Data_2012[-folds[[i]],]
  
  # build the model
  logit_4 <- update(logit_3, . ~ . + ind2_cat + growth_rate*ind2_cat + net_profit_margin*ind2_cat + current_ratio_cat*ind2_cat + equity_to_assets_cat*ind2_cat + fixed_assets_ratio*ind2_cat + intang_assets_ratio*ind2_cat + emp_avg*ind2_cat + emp_growth_rate*ind2_cat +  personnel_exp_per_emp*ind2_cat +  fixed_assets_growth_rate_cat*ind2_cat + intang_assets_growth_rate_cat*ind2_cat + personnel_exp_per_emp_growth_rate_cat*ind2_cat, data = train)
  
  # predict the test set
  pred <- predict(logit_4, test, type = "response")
  
  # calculate RMSE
  rmse <- sqrt(mean((pred - test$fast_growing)^2))
  
  # calculate AUC
  auc <- auc(test$fast_growing, pred)
  
  # store the results
  results[[i]] <- rmse
  
  results_auc[[i]] <- auc
}

# calculate the average RMSE
RMSE_Logit_4 <- sqrt(mean(unlist(results)^2))

# calculate the average AUC
AUC_Logit_4 <- mean(unlist(results_auc))

## Model 5

results <- list()
results_auc <- list()

for (i in 1:5) {
  # get the training set
  train <- Data_2012[folds[[i]],]
  
  # get the test set
  test <- Data_2012[-folds[[i]],]
  
  # build the model
  logit_5 <- update(logit_4, . ~ . + ceo_count + female + inoffice_days + urban_m + region_m + age + I(age^2) + ceo_age_young + ceo_age_middle + gender_m + origin_m + foreign_management + inventory_to_sales + flag_inventories_problem + inventory_to_sales_flag, data = train)
  
  # predict the test set
  pred <- predict(logit_5, test, type = "response")
  
  # calculate RMSE
  rmse <- sqrt(mean((pred - test$fast_growing)^2))
  
  # calculate AUC
  auc <- auc(test$fast_growing, pred)
  
  # store the results
  results[[i]] <- rmse
  
  results_auc[[i]] <- auc
}

# calculate the average RMSE
RMSE_Logit_5 <- sqrt(mean(unlist(results)^2))

# calculate the average AUC
AUC_Logit_5 <- mean(unlist(results_auc))

#### LASSO Logistic Regression
### Create sets of variable to use in LASSO.
interaction_terms <- c("ind2_cat*growth_rate", "ind2_cat*ceo_count", "ind2_cat*female", "ind2_cat*inoffice_days", "ind2_cat*urban_m", "ind2_cat*region_m", "ind2_cat*age", "ind2_cat*ceo_age_young", "ind2_cat*ceo_age_middle", 
                       "ind2_cat*gender_m","ind2_cat*origin_m", "ind2_cat*foreign_management", "ind2_cat*inventory_to_sales",  "ind2_cat*net_profit_margin", "ind2_cat*current_ratio_cat", "ind2_cat*equity_to_assets_cat", "ind2_cat*fixed_assets_ratio", "ind2_cat*intang_assets_ratio", "ind2_cat*emp_avg", "ind2_cat*emp_growth_rate", "ind2_cat*personnel_exp_per_emp", "ind2_cat*fixed_assets_growth_rate_cat", "ind2_cat*intang_assets_growth_rate_cat", "ind2_cat*personnel_exp_per_emp_growth_rate_cat")

# Original variable
LASSOVAR <- colnames(Data_2012)

# exclude ind
LASSOVAR <- LASSOVAR[LASSOVAR != "ind"]

# exclude fast_growing
LASSOVAR <- LASSOVAR[LASSOVAR != "fast_growing"]

## Combine
Lasso_vars <- c(LASSOVAR, interaction_terms)

### Recode fast_growing as factor.
Data_2012$fast_growing <- as.factor(Data_2012$fast_growing)

Data_2012_hold_out$fast_growing <- as.factor(Data_2012_hold_out$fast_growing)

## Relabel the levels of fast_growing
levels(Data_2012$fast_growing) <- c("No", "Yes")

levels(Data_2012_hold_out$fast_growing) <- c("No", "Yes")

### Fit LASSO logistic regression model using glmnet 5 fold cross validation.
set.seed(20231210)

# Set up my own summary function.
RMSE_sum <- function(data, lev = NULL, model = NULL) {
  probs <- data[, lev[1]]
  c(rmse = RMSE(pred = probs,
                obs = ifelse(data$obs == lev[1], 1, 0)))
}

# Define your custom summary function
customSummary <- function(data, lev = NULL, model = NULL) {
  # Calculate metrics using twoClassSummary
  default_metrics <- twoClassSummary(data, lev, model)
  
  # Calculate your custom metric (RMSE_sum)
  additional_metric <- RMSE_sum(data, lev, model)
  
  # Combine the metrics into a vector
  c(default_metrics, additional_metric)
}


# setting up the control
control <- trainControl(method = "cv", number = 5, summaryFunction = customSummary, classProbs = TRUE)

# Fit the model

lambda <- 10^seq(-0.1, -4, length = 20)
grid <- expand.grid("alpha" = 1, lambda = lambda)

logit_lasso_model <- train(formula(paste0("fast_growing ~", paste0(Lasso_vars, collapse = " + "))),
                           data = Data_2012,
                           method = "glmnet",
                           preProcess = c("center", "scale"),
                           family = "binomial",
                           trControl = control,
                           tuneGrid = grid,
                           na.action=na.exclude,  metric = "ROC")


# Get the optimal value of lambda
lambda_optimal <- logit_lasso_model$bestTune$lambda

## Report the optimal lambda.
cat("Optimal lambda:", lambda_optimal, "\n")

## get the MAE (in 5 test set).
RMSE_lasso_5f <- logit_lasso_model$resample$rmse

## RMSE
RMSE_lasso <- sqrt(mean(RMSE_lasso_5f^2))

## Extract AUC from the model.
AUC_lasso <- mean(logit_lasso_model$resample$ROC)

#### Random Forest

## Setting up the random forest
# specifying the range of values for the tuning parameters

mtry <- seq(5, 14, by = 1)

# Setting up the grid
grid <- expand.grid(mtry = mtry)

# setting up the control
control <- trainControl(method = "cv", number = 5, summaryFunction = customSummary, classProbs = TRUE)

# tuning the model
set.seed(20231210)

# I set ntree = 150 for the sake of computation time.
model_RF <- train(fast_growing ~ ., data = Data_2012[,!names(Data_2012) %in% c("ind")], method = "rf", trControl = control, tuneGrid = grid, ntree = 150, metric = "ROC")

## get the optimal value of mtry
mtry_optimal <- model_RF$bestTune$mtry

# print
cat("Optimal mtry:", mtry_optimal, "\n")

### Extracting RMSE and AUC from the model.
RMSE_RF_5f <- model_RF$resample$rmse

## RMSE
RMSE_RF <- sqrt(mean(RMSE_RF_5f^2))

## AUC
AUC_RF <- mean(model_RF$resample$ROC)

### Compiling the results
## Compile the results.
# model name
Model_Name <- c("Logit 1", "Logit 2", "Logit 3", "Logit 4", "Logit 5")

# performance
RMSE_Logit <- c(RMSE_Logit_1, RMSE_Logit_2, RMSE_Logit_3, RMSE_Logit_4, RMSE_Logit_5)

AUC_Logit <- c(AUC_Logit_1, AUC_Logit_2, AUC_Logit_3, AUC_Logit_4, AUC_Logit_5)

# Make a data frame.
Logit_results <- data.frame(Model_Name, RMSE_Logit, AUC_Logit)


# Show the results. Use kableExtra to make the table look nicer.

kable(Logit_results, digits = 4, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))

### Compare results of Logit 2, LASSO, and Random Forest.
# model name
Model_Name <- c("Logit 2", "LASSO", "Random Forest")

# performance
RMSE <- c(RMSE_Logit_2, RMSE_lasso, RMSE_RF)

AUC <- c(AUC_Logit_2, AUC_lasso, AUC_RF)

# Make a data frame.
All_results <- data.frame(Model_Name, RMSE, AUC)

# Show the results. Use kableExtra to make the table look nicer.

kable(All_results, digits = 4, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))

#### Part 2: Minimizing expected loss.
### Find the optimal threshold that minimizes the expected loss.

# cost of false positive
cost_fp <- 130

# cost of false negative
cost_fn <- 100

# The range of threshold
thresholds <- seq(0.45, 0.95, by = 0.025)

## Logit 2

num_folds <- 5

cv_losses <- numeric(length(thresholds))

# Loop over the thresholds
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  total_loss <- 0
  
  # Perform cross-validation
  for (fold in 1:num_folds) {
    # Split the data into training and testing sets
    train_data <- Data_2012[-folds[[fold]], ]
    test_data <- Data_2012[folds[[fold]], ]
    
    # Fit the logistic regression model on the training data
    # (assuming 'target_variable' is your dependent variable)
    logit_model <- update(logit_1, . ~ . + net_profit_margin + current_ratio_cat + equity_to_assets_cat + flag_asset_problem + flag_curr_liab_problem + net_profit_margin_flag_low + net_profit_margin_flag_high, data = train_data)
    
    # Make predictions on the testing data
    predicted_probs <- predict(logit_model, newdata = test_data, type = "response")
    
    ## Classify the observations based on the threshold
    predicted_classes <- ifelse(predicted_probs > threshold, 1, 0)
    
    ## Calculate expected loss for the current threshold
    # Create dummy: true positive.
    True_pos <- ifelse(test_data$fast_growing == "Yes" & predicted_classes == 1, 1, 0)
    
    # Create dummy: false positive.
    False_pos <- ifelse(test_data$fast_growing == "No" & predicted_classes == 1, 1, 0)
    
    # Create dummy: true negative.
    True_neg <- ifelse(test_data$fast_growing == "No" & predicted_classes == 0, 1, 0)
    
    # Create dummy: false negative.
    False_neg <- ifelse(test_data$fast_growing == "Yes" & predicted_classes == 0, 1, 0)
    
    # Calculate the P(False Positive)
    P_fp <- sum(False_pos) / length(predicted_classes)
    
    # Calculate the P(False Negative)
    P_fn <- sum(False_neg) / length(predicted_classes)
    
    # Calculate the expected loss
    total_loss <- (cost_fp * P_fp) + (cost_fn * P_fn)
    
    # Calculate average loss across folds for the current threshold
    cv_losses[i] <- total_loss / num_folds
  }
}

# Find the threshold that minimizes the expected loss
optimal_threshold_logit <- thresholds[which.min(cv_losses)]

# Store the minimum expected loss
min_loss_logit <- min(cv_losses)

# Print the optimal threshold for logit 2
cat("Optimal Threshold for Logit 2:", optimal_threshold_logit, "\n")

## LASSO
tuned_logit_lasso_model <- logit_lasso_model$finalModel

cv_losses <- numeric(length(thresholds))

# Loop over the thresholds
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  total_loss <- 0
  
  # Perform cross-validation
  for (fold in 1:num_folds) {
    # Split the data into training and testing sets
    train_data <- Data_2012[-folds[[fold]], ]
    test_data <- Data_2012[folds[[fold]], ]
    
    
    # Make predictions on the testing data
    newx_lasso = model.matrix(formula(paste0("fast_growing ~", paste0(Lasso_vars, collapse = " + "))), test_data)[, -1]
    
    predicted_probs <- predict(tuned_logit_lasso_model, newdata = test_data, type = "response", newx = newx_lasso, s = lambda_optimal)
    
    ## Classify the observations based on the threshold
    predicted_classes <- ifelse(predicted_probs > threshold, 1, 0)
    
    ## Calculate expected loss for the current threshold
    # Create dummy: true positive.
    True_pos <- ifelse(test_data$fast_growing == "Yes" & predicted_classes == 1, 1, 0)
    
    # Create dummy: false positive.
    False_pos <- ifelse(test_data$fast_growing == "No" & predicted_classes == 1, 1, 0)
    
    # Create dummy: true negative.
    True_neg <- ifelse(test_data$fast_growing == "No" & predicted_classes == 0, 1, 0)
    
    # Create dummy: false negative.
    False_neg <- ifelse(test_data$fast_growing == "Yes" & predicted_classes == 0, 1, 0)
    
    # Calculate the P(False Positive)
    P_fp <- sum(False_pos) / length(predicted_classes)
    
    # Calculate the P(False Negative)
    P_fn <- sum(False_neg) / length(predicted_classes)
    
    # Calculate the expected loss
    total_loss <- (cost_fp * P_fp) + (cost_fn * P_fn)
    
    # Calculate average loss across folds for the current threshold
    cv_losses[i] <- total_loss / num_folds
  }
}

# Find the threshold that minimizes the expected loss
optimal_threshold_lasso <- thresholds[which.min(cv_losses)]

# Store the minimum expected loss
min_loss_lasso <- min(cv_losses)

# Print the optimal threshold for LASSO
cat("Optimal Threshold for LASSO:", optimal_threshold_lasso, "\n")

## Random Forest

cv_losses <- numeric(length(thresholds))

# Loop over the thresholds
for (i in seq_along(thresholds)) {
  threshold <- thresholds[i]
  total_loss <- 0
  
  # Perform cross-validation
  for (fold in 1:num_folds) {
    # Split the data into training and testing sets
    train_data <- Data_2012[-folds[[fold]], ]
    test_data <- Data_2012[folds[[fold]], ]
    
    # Fit the random forest model on the training data
    
    predictors <- train_data[, -which(names(train_data) %in% c("fast_growing"))]
    
    model_RF_loss <- randomForest(x = predictors, y = train_data$fast_growing, ntree = 150, mtry = mtry_optimal)
    
    # Make predictions on the testing data
    predicted_probs <- predict(model_RF_loss, newdata = test_data, type = "prob")
    
    ## Classify the observations based on the threshold
    predicted_classes <- ifelse(predicted_probs > threshold, 1, 0)
    
    ## Calculate expected loss for the current threshold
    # Create dummy: true positive.
    True_pos <- ifelse(test_data$fast_growing == "Yes" & predicted_classes == 1, 1, 0)
    
    # Create dummy: false positive.
    False_pos <- ifelse(test_data$fast_growing == "No" & predicted_classes == 1, 1, 0)
    
    # Create dummy: true negative.
    True_neg <- ifelse(test_data$fast_growing == "No" & predicted_classes == 0, 1, 0)
    
    # Create dummy: false negative.
    False_neg <- ifelse(test_data$fast_growing == "Yes" & predicted_classes == 0, 1, 0)
    
    # Calculate the P(False Positive)
    P_fp <- sum(False_pos) / length(predicted_classes)
    
    # Calculate the P(False Negative)
    P_fn <- sum(False_neg) / length(predicted_classes)
    
    # Calculate the expected loss
    total_loss <- (cost_fp * P_fp) + (cost_fn * P_fn)
    
    # Calculate average loss across folds for the current threshold
    cv_losses[i] <- total_loss / num_folds
  }
}

# Find the threshold that minimizes the expected loss
optimal_threshold_RF <- thresholds[which.min(cv_losses)]

# Store the minimum expected loss
min_loss_RF <- min(cv_losses)

# Print the optimal threshold for Random Forest
cat("Optimal Threshold for Random Forest:", optimal_threshold_RF, "\n")

### Compiling the results into a table.

# Create a data frame with the results
results_loss <- data.frame(
  Model = c("Logit 2", "LASSO", "Random Forest"),
  Optimal_Threshold = c(optimal_threshold_logit, optimal_threshold_lasso, optimal_threshold_RF),
  Expected_Loss = c(min_loss_logit, min_loss_lasso, min_loss_RF)
)

# Print the results
kable(results_loss, digits = 4, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))

#### Part 3: Confusion Matrix
## Re-fit the models on the entire work set
# Fit the logit model

best_logit_model <- update(logit_1, . ~ . + net_profit_margin + current_ratio_cat + equity_to_assets_cat + flag_asset_problem + flag_curr_liab_problem + net_profit_margin_flag_low + net_profit_margin_flag_high, data = Data_2012)

# Fit the LASSO model 

best_lasso_model = tuned_logit_lasso_model

# Fit the random forest model
predictors <- Data_2012[, -which(names(Data_2012) %in% c("fast_growing"))]

best_RF_model <- randomForest(x = predictors, y = Data_2012$fast_growing, ntree = 150, mtry = mtry_optimal)

## Make predictions on the hold-out set.

# Make predictions using the logit model
Data_2012_hold_out$predicted_probs_logit <- predict(best_logit_model, newdata = Data_2012_hold_out, type = "response")

# Make predictions using the LASSO model

newx_lasso_hold_out = model.matrix(formula(paste0("fast_growing ~", paste0(Lasso_vars, collapse = " + "))), Data_2012_hold_out)[, -1]

Data_2012_hold_out$predicted_probs_lasso <- predict(best_lasso_model, newdata = Data_2012_hold_out, type = "response", s = lambda_optimal, newx = newx_lasso_hold_out)

# Make predictions using the random forest model
Data_2012_hold_out$predicted_probs_RF <- predict(best_RF_model, newdata = Data_2012_hold_out, type = "prob")[, 2]

### Get number of true positives, false positives, true negatives, and false negatives for each model, using the optimal threshold of each model.

## Logit
# Create dummy: true positive_logit.
Data_2012_hold_out$True_pos_logit <- ifelse(Data_2012_hold_out$fast_growing == "Yes" & Data_2012_hold_out$predicted_probs_logit > optimal_threshold_logit, 1, 0)

# Create dummy: false positive_logit.
Data_2012_hold_out$False_pos_logit <- ifelse(Data_2012_hold_out$fast_growing == "No" & Data_2012_hold_out$predicted_probs_logit > optimal_threshold_logit, 1, 0)

# Create dummy: true negative_logit.
Data_2012_hold_out$True_neg_logit <- ifelse(Data_2012_hold_out$fast_growing == "No" & Data_2012_hold_out$predicted_probs_logit <= optimal_threshold_logit, 1, 0)

# Create dummy: false negative_logit.
Data_2012_hold_out$False_neg_logit <- ifelse(Data_2012_hold_out$fast_growing == "Yes" & Data_2012_hold_out$predicted_probs_logit <= optimal_threshold_logit, 1, 0)

## LASSO
# Create dummy: true positive_lasso.
Data_2012_hold_out$True_pos_lasso <- ifelse(Data_2012_hold_out$fast_growing == "Yes" & Data_2012_hold_out$predicted_probs_lasso > optimal_threshold_lasso, 1, 0)

# Create dummy: false positive_lasso.
Data_2012_hold_out$False_pos_lasso <- ifelse(Data_2012_hold_out$fast_growing == "No" & Data_2012_hold_out$predicted_probs_lasso > optimal_threshold_lasso, 1, 0)

# Create dummy: true negative_lasso.
Data_2012_hold_out$True_neg_lasso <- ifelse(Data_2012_hold_out$fast_growing == "No" & Data_2012_hold_out$predicted_probs_lasso <= optimal_threshold_lasso, 1, 0)

# Create dummy: false negative_lasso.
Data_2012_hold_out$False_neg_lasso <- ifelse(Data_2012_hold_out$fast_growing == "Yes" & Data_2012_hold_out$predicted_probs_lasso <= optimal_threshold_lasso, 1, 0)

## Random Forest
# Create dummy: true positive_RF.
Data_2012_hold_out$True_pos_RF <- ifelse(Data_2012_hold_out$fast_growing == "Yes" & Data_2012_hold_out$predicted_probs_RF > optimal_threshold_RF, 1, 0)

# Create dummy: false positive_RF.
Data_2012_hold_out$False_pos_RF <- ifelse(Data_2012_hold_out$fast_growing == "No" & Data_2012_hold_out$predicted_probs_RF > optimal_threshold_RF, 1, 0)

# Create dummy: true negative_RF.
Data_2012_hold_out$True_neg_RF <- ifelse(Data_2012_hold_out$fast_growing == "No" & Data_2012_hold_out$predicted_probs_RF <= optimal_threshold_RF, 1, 0)

# Create dummy: false negative_RF.
Data_2012_hold_out$False_neg_RF <- ifelse(Data_2012_hold_out$fast_growing == "Yes" & Data_2012_hold_out$predicted_probs_RF <= optimal_threshold_RF, 1, 0)

### Calculate each element of the confusion matrix for each model.

## Logit
# Calculate number of true positives_logit.
true_pos_logit <- sum(Data_2012_hold_out$True_pos_logit)

# Calculate number of false positives_logit.
false_pos_logit <- sum(Data_2012_hold_out$False_pos_logit)

# Calculate number of true negatives_logit.
true_neg_logit <- sum(Data_2012_hold_out$True_neg_logit)

# Calculate number of false negatives_logit.
false_neg_logit <- sum(Data_2012_hold_out$False_neg_logit)

# Calculate total number of positives_logit.
total_pos_logit <- true_pos_logit + false_neg_logit

# Calculate total number of negatives_logit.
total_neg_logit <- true_neg_logit + false_pos_logit

# calculate total number of predicted positives_logit.
total_predicted_pos_logit <- true_pos_logit + false_pos_logit

# calculate total number of predicted negatives_logit.
total_predicted_neg_logit <- true_neg_logit + false_neg_logit

# calculate total number of observations.
total_obs_logit <- total_pos_logit + total_neg_logit

## LASSO
# Calculate number of true positives_lasso.
true_pos_lasso <- sum(Data_2012_hold_out$True_pos_lasso)

# Calculate number of false positives_lasso.
false_pos_lasso <- sum(Data_2012_hold_out$False_pos_lasso)

# Calculate number of true negatives_lasso.
true_neg_lasso <- sum(Data_2012_hold_out$True_neg_lasso)

# Calculate number of false negatives_lasso.
false_neg_lasso <- sum(Data_2012_hold_out$False_neg_lasso)

# Calculate total number of positives_lasso.
total_pos_lasso <- true_pos_lasso + false_neg_lasso

# Calculate total number of negatives_lasso.
total_neg_lasso <- true_neg_lasso + false_pos_lasso

# calculate total number of predicted positives_lasso.
total_predicted_pos_lasso <- true_pos_lasso + false_pos_lasso

# calculate total number of predicted negatives_lasso.
total_predicted_neg_lasso <- true_neg_lasso + false_neg_lasso

# calculate total number of observations.
total_obs_lasso <- total_pos_lasso + total_neg_lasso

## Random Forest
# Calculate number of true positives_RF.
true_pos_RF <- sum(Data_2012_hold_out$True_pos_RF)

# Calculate number of false positives_RF.
false_pos_RF <- sum(Data_2012_hold_out$False_pos_RF)

# Calculate number of true negatives_RF.
true_neg_RF <- sum(Data_2012_hold_out$True_neg_RF)

# Calculate number of false negatives_RF.
false_neg_RF <- sum(Data_2012_hold_out$False_neg_RF)

# Calculate total number of positives_RF.
total_pos_RF <- true_pos_RF + false_neg_RF

# Calculate total number of negatives_RF.
total_neg_RF <- true_neg_RF + false_pos_RF

# calculate total number of predicted positives_RF.
total_predicted_pos_RF <- true_pos_RF + false_pos_RF

# calculate total number of predicted negatives_RF.
total_predicted_neg_RF <- true_neg_RF + false_neg_RF

# calculate total number of observations.
total_obs_RF <- total_pos_RF + total_neg_RF

```


```{r}
### Make Confusion Matrix for Logit. (3 colunms: Actual Fast Growing, Actual Not Fast Growing, Total; 3 rows: Predicted Fast Growing, Predicted Not Fast Growing, Total)

# Create matrix for confusion matrix_logit.
confusion_matrix_logit <- matrix(c(true_pos_logit, false_pos_logit, total_predicted_pos_logit,
                                   false_neg_logit, true_neg_logit, total_predicted_neg_logit,
                                   total_pos_logit, total_neg_logit, total_obs_logit),
                                 nrow = 3, ncol = 3, byrow = TRUE)

# Divide each element in confusion matrix_lasso by total number of observations and x 100 to get percentage.
confusion_matrix_logit <- (confusion_matrix_logit / total_obs_logit) * 100

# convert matrix to dataframe.
confusion_matrix_logit <- as.data.frame(confusion_matrix_logit)

# Add column names to confusion matrix_logit.
colnames(confusion_matrix_logit) <- c("Actual Fast Growing", "Actual Not Fast Growing", "Total")

# Add row names to confusion matrix_logit.
rownames(confusion_matrix_logit) <- c("Predicted Fast Growing", "Predicted Not Fast Growing", "Total")

# Print confusion matrix_logit.
kable(confusion_matrix_logit, digits = 2, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))

## Examine the top values of predicted probabilities for the fast growing class.
# Create dataframe of predicted probabilities for fast growing class.
predicted_probs_logit <- Data_2012_hold_out %>%
  select(fast_growing, predicted_probs_logit) %>%
  arrange(desc(predicted_probs_logit))


kable(head(predicted_probs_logit , 5))

# We can see that all of predicted probabilities fall below the optimal threshold. 

### Make Confusion Matrix for LASSO. (3 colunms: Actual Fast Growing, Actual Not Fast Growing, Total; 3 rows: Predicted Fast Growing, Predicted Not Fast Growing, Total)

# Create matrix for confusion matrix_lasso.
confusion_matrix_lasso <- matrix(c(true_pos_lasso, false_pos_lasso, total_predicted_pos_lasso,
                                   false_neg_lasso, true_neg_lasso, total_predicted_neg_lasso,
                                   total_pos_lasso, total_neg_lasso, total_obs_lasso),
                                 nrow = 3, ncol = 3, byrow = TRUE)

# Divide each element in confusion matrix_lasso by total number of observations and x 100 to get percentage.
confusion_matrix_lasso <- confusion_matrix_lasso / total_obs_lasso * 100

# convert matrix to dataframe.
confusion_matrix_lasso <- as.data.frame(confusion_matrix_lasso)

# Add column names to confusion matrix_lasso.
colnames(confusion_matrix_lasso) <- c("Actual Fast Growing", "Actual Not Fast Growing", "Total")

# Add row names to confusion matrix_lasso.
rownames(confusion_matrix_lasso) <- c("Predicted Fast Growing", "Predicted Not Fast Growing", "Total")

# Print confusion matrix_lasso.
kable(confusion_matrix_lasso, digits = 2, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))

## Examine the top values of predicted probabilities for the fast growing class.
# Create dataframe of predicted probabilities for fast growing class.
predicted_probs_lasso <- Data_2012_hold_out %>%
  select(fast_growing, predicted_probs_lasso) %>%
  arrange(desc(predicted_probs_lasso))


kable(head(predicted_probs_lasso , 5))

### Make Confusion Matrix for Random Forest. (3 colunms: Actual Fast Growing, Actual Not Fast Growing, Total; 3 rows: Predicted Fast Growing, Predicted Not Fast Growing, Total)

# Create matrix for confusion matrix_RF.
confusion_matrix_RF <- matrix(c(true_pos_RF, false_pos_RF, total_predicted_pos_RF,
                                false_neg_RF, true_neg_RF, total_predicted_neg_RF,
                                total_pos_RF, total_neg_RF, total_obs_RF),
                              nrow = 3, ncol = 3, byrow = TRUE)

# Divide each element in confusion matrix_lasso by total number of observations and x 100 to get percentage.
confusion_matrix_RF <- confusion_matrix_RF / total_obs_RF * 100

# convert matrix to dataframe.
confusion_matrix_RF <- as.data.frame(confusion_matrix_RF)

# Add column names to confusion matrix_RF.
colnames(confusion_matrix_RF) <- c("Actual Fast Growing", "Actual Not Fast Growing", "Total")

# Add row names to confusion matrix_RF.
rownames(confusion_matrix_RF) <- c("Predicted Fast Growing", "Predicted Not Fast Growing", "Total")

# Print confusion matrix_RF.
kable(confusion_matrix_RF, digits = 2, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))

## Examine the top values of predicted probabilities for the fast growing class.
# Create dataframe of predicted probabilities for fast growing class.
predicted_probs_RF <- Data_2012_hold_out %>%
  select(fast_growing, predicted_probs_RF) %>%
  arrange(desc(predicted_probs_RF))


kable(head(predicted_probs_RF , 5))

### Calculate the expected loss for each model on the hold-out set.

## Logit
# False positive rate.
false_pos_rate_logit <- false_pos_logit / total_obs_logit

# False negative rate.
false_neg_rate_logit <- false_neg_logit / total_obs_logit

# Expected loss.
expected_loss_logit <- false_pos_rate_logit * cost_fp + false_neg_rate_logit * cost_fn

## LASSO
# False positive rate.
false_pos_rate_lasso <- false_pos_lasso / total_obs_lasso

# False negative rate.
false_neg_rate_lasso <- false_neg_lasso / total_obs_lasso

# Expected loss.
expected_loss_lasso <- false_pos_rate_lasso * cost_fp + false_neg_rate_lasso * cost_fn

## Random Forest
# False positive rate.
false_pos_rate_RF <- false_pos_RF / total_obs_RF

# False negative rate.
false_neg_rate_RF <- false_neg_RF / total_obs_RF

# Expected loss.
expected_loss_RF <- false_pos_rate_RF * cost_fp + false_neg_rate_RF * cost_fn

### Obtain RMSE and AUC in the hold-out set.
## Logit

# RMSE.
RMSE_logit_hold_out <- RMSE((as.numeric(Data_2012_hold_out$fast_growing)-1), Data_2012_hold_out$predicted_probs_logit)

# AUC.
AUC_logit_hold_out <- auc(roc(Data_2012_hold_out$fast_growing, Data_2012_hold_out$predicted_probs_logit))

## LASSO

# RMSE.
RMSE_lasso_hold_out <- RMSE((as.numeric(Data_2012_hold_out$fast_growing)-1), Data_2012_hold_out$predicted_probs_lasso)

# AUC.
AUC_lasso_hold_out <- auc(roc(Data_2012_hold_out$fast_growing, Data_2012_hold_out$predicted_probs_lasso))

## Random Forest

# RMSE.
RMSE_RF_hold_out <- RMSE((as.numeric(Data_2012_hold_out$fast_growing)-1), Data_2012_hold_out$predicted_probs_RF)

# AUC.
AUC_RF_hold_out <- auc(roc(Data_2012_hold_out$fast_growing, Data_2012_hold_out$predicted_probs_RF))

### Compile the results of RMSE, AUC, and expected loss into a table.

# Create dataframe of RMSE, AUC, and expected loss.
RMSE_AUC_expected_loss <- data.frame(model = c("Logit", "LASSO", "Random Forest"),
                                     RMSE = c(RMSE_logit_hold_out, RMSE_lasso_hold_out, RMSE_RF_hold_out),
                                     AUC = c(AUC_logit_hold_out, AUC_lasso_hold_out, AUC_RF_hold_out),
                                     expected_loss = c(expected_loss_logit, expected_loss_lasso, expected_loss_RF))

# Print the table
kable(RMSE_AUC_expected_loss, digits = 4, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))


###### Task 2
Data_2012 %>% count(ind)

## Seperate both work and hold-out sets.

# Create work set for manufacturing sector.
Data_2012_manu <- Data_2012 %>%
  filter(ind == 2)

# Create hold-out set for manufacturing sector.
Data_2012_hold_out_manu <- Data_2012_hold_out %>%
  filter(ind == 2)

# Create work set for service sector.
Data_2012_service <- Data_2012 %>%
  filter(ind == 3)

# Create hold-out set for service sector.
Data_2012_hold_out_service <- Data_2012_hold_out %>%
  filter(ind == 3)

#### Let's do stuff with manufacturing sector first.
### We already predict. 

## Calculate RMSE.
# Calculate RMSE for logit.
RMSE_logit_manu <- RMSE((as.numeric(Data_2012_hold_out_manu$fast_growing)-1), Data_2012_hold_out_manu$predicted_probs_logit)

## Calculate AUC
# Calculate AUC for logit.
AUC_logit_manu <- auc(Data_2012_hold_out_manu$fast_growing, Data_2012_hold_out_manu$predicted_probs_logit)

### Get elements for confusion matrix.
# Get true positives for logit.
true_pos_logit_manu <- sum(Data_2012_hold_out_manu$True_pos_logit)

# Get false positives for logit.
false_pos_logit_manu <- sum(Data_2012_hold_out_manu$False_pos_logit)

# Get false negatives for logit.
false_neg_logit_manu <- sum(Data_2012_hold_out_manu$False_neg_logit)

# Get true negatives for logit.
true_neg_logit_manu <- sum(Data_2012_hold_out_manu$True_neg_logit)

# Get total number of observations for logit.
total_obs_logit_manu <- true_pos_logit_manu + false_pos_logit_manu + false_neg_logit_manu + true_neg_logit_manu

# Get total positives for logit.
total_pos_logit_manu <- true_pos_logit_manu + false_neg_logit_manu

# Get total negatives for logit.
total_neg_logit_manu <- true_neg_logit_manu + false_pos_logit_manu

# Get total predicted positives for logit.
total_pred_pos_logit_manu <- true_pos_logit_manu + false_pos_logit_manu

# Get total predicted negatives for logit.
total_pred_neg_logit_manu <- true_neg_logit_manu + false_neg_logit_manu

# Get False positive rate for logit.
false_pos_rate_logit_manu <- false_pos_logit_manu / total_obs_logit_manu

# Get False negative rate for logit.
false_neg_rate_logit_manu <- false_neg_logit_manu / total_obs_logit_manu

# Get expected loss for logit.
expected_loss_logit_manu <- false_pos_rate_logit_manu * cost_fp + false_neg_rate_logit_manu * cost_fn

### Create Summary Table for Manufacturing Sector. (RMSE, AUC, Expected Loss) (1 rows x 3 columns)

Summary_logit_manu <- data.frame(RMSE = RMSE_logit_manu, AUC = AUC_logit_manu, Expected_Loss = expected_loss_logit_manu)

# Add row names.
rownames(Summary_logit_manu) <- "Manufacturing Sector"

#### Do the same for service sector.
### We already predict.

## Calculate RMSE.
# Calculate RMSE for logit.
RMSE_logit_service <- RMSE((as.numeric(Data_2012_hold_out_service$fast_growing)-1), Data_2012_hold_out_service$predicted_probs_logit)

## Calculate AUC
# Calculate AUC for logit.

AUC_logit_service <- auc(Data_2012_hold_out_service$fast_growing, Data_2012_hold_out_service$predicted_probs_logit)

### Get elements for confusion matrix.
# Get true positives for logit.
true_pos_logit_service <- sum(Data_2012_hold_out_service$True_pos_logit)

# Get false positives for logit.
false_pos_logit_service <- sum(Data_2012_hold_out_service$False_pos_logit)

# Get false negatives for logit.
false_neg_logit_service <- sum(Data_2012_hold_out_service$False_neg_logit)

# Get true negatives for logit.
true_neg_logit_service <- sum(Data_2012_hold_out_service$True_neg_logit)

# Get total number of observations for logit.
total_obs_logit_service <- true_pos_logit_service + false_pos_logit_service + false_neg_logit_service + true_neg_logit_service

# Get total positives for logit.
total_pos_logit_service <- true_pos_logit_service + false_neg_logit_service

# Get total negatives for logit.
total_neg_logit_service <- true_neg_logit_service + false_pos_logit_service

# Get total predicted positives for logit.
total_pred_pos_logit_service <- true_pos_logit_service + false_pos_logit_service

# Get total predicted negatives for logit.
total_pred_neg_logit_service <- true_neg_logit_service + false_neg_logit_service

# Get False positive rate for logit.
false_pos_rate_logit_service <- false_pos_logit_service / total_obs_logit_service

# Get False negative rate for logit.
false_neg_rate_logit_service <- false_neg_logit_service / total_obs_logit_service

# Get expected loss for logit.
expected_loss_logit_service <- false_pos_rate_logit_service * cost_fp + false_neg_rate_logit_service * cost_fn

### Create Summary Table for Service Sector. (RMSE, AUC, Expected Loss) (1 rows x 3 columns)

Summary_logit_service <- data.frame(RMSE = RMSE_logit_service, AUC = AUC_logit_service, Expected_Loss = expected_loss_logit_service)

# Add row names.
rownames(Summary_logit_service) <- "Service Sector"

## rbind the two summary tables.
Summary_logit <- rbind(Summary_logit_manu, Summary_logit_service)

# make AUC numeric.
Summary_logit$AUC <- as.numeric(Summary_logit$AUC)

# Print
# Print the results
kable(Summary_logit, digits = 4, align = "c") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "bordered"))