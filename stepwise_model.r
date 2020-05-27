setwd("~/Winter Semester - BDSA/PROG8420 - Programming for Big Data/Assignments/Final Assignment")

library(readr)
dataset <- read_csv("data.csv")
dataset = dataset[,-c(1:2)]

library(MASS)

full_model = lm(Money ~., data = dataset)

# Stepwise Selection Model
both_model = stepAIC(full_model, direction = 'both', trace = FALSE)
summary(both_model)

golf_lm = lm(Money ~., data = dataset)
summary(golf_lm)

bck_golf = step(golf_lm, direction = 'backward', details = TRUE)
summary(bck_golf)
