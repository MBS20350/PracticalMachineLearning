---
title: "JHS Practical Machine Learning Project"
output: 
  html_document: 
    keep_md: true
---



<center>
# Evaluating Exercise with Machine Learning  
### by Mark Sucato
</center>

### Executive Summary

The HAR *WLE* dataset contains on-body sensor information from six participants 
performing 10 repetitions of a unilateral dumbell biceps curl in five different 
but specific manners.  A stacked ensemble of three different classification 
tree-based models trained on a training set of 13,737 observations predicted a 
validation set of 4,127 observations with 99.98% accuracy. 

### Project objective and data

**Objective**:  predict the manner of exercise for twenty observations drawn 
from the Human Activity Recognition *Weight Lifting Exercises* dataset. 

The HAR *WLE* dataset contains on-body sensor information from six participants 
performing 10 repetitions of a unilateral dumbell biceps curl in five 
different but specific manners 
(http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).
Each observation is a time-stamped capture of sensor data; one complete 
repetition by a given participant includes multiple sequential observations. 
The 2013 *Qualitative Activity Recognition of Weight Lifting Exercises* HAR paper 
by Velloso et al, available at the same website, analyzed time slices of 
sequential data to assess repetitions ^1 . Because the provided test set for this project only contains 
 single-observation data excerpted from the greater dataset, sequential data 
analysis is not feasible for this project. 

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. 
Qualitative Activity Recognition of Weight Lifting Exercises. 
Proceedings of 4th International Conference in Cooperation with SIGCHI 
(Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.    

### Exploratory Data Analysis

The provided training set includes 19,622 observations of 160 variables.  100 
of these variables, all seemingly summary calculations from the raw sensor 
data, are almost entirely missing.  Because this project will not rely on any 
sequential analysis, the time stamp observations, row labels, and movement 
window indicators are also unnecessary.  A *skim* summary of the parsed data 
is provided below.     


```r
library(tidyverse)
library(caret)
library(skimr)
library(doParallel)

training <- read_csv("pml-training.csv")
testing <- read_csv("pml-testing.csv")

training <- training %>%
  select(where(~mean(is.na(.))< 0.9)) %>%
  select(-c(X1, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window)) %>%
  mutate(classe=factor(classe), user_name = factor(user_name))
fix_windows_histograms()   # skimr package utility to fix histogram printing on Windows
skim(training)
```


<table style='width: auto;'
        class='table table-condensed'>
<caption>Data summary</caption>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:left;">   </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Name </td>
   <td style="text-align:left;"> training </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Number of rows </td>
   <td style="text-align:left;"> 19622 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Number of columns </td>
   <td style="text-align:left;"> 54 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> _______________________ </td>
   <td style="text-align:left;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Column type frequency: </td>
   <td style="text-align:left;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> factor </td>
   <td style="text-align:left;"> 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> numeric </td>
   <td style="text-align:left;"> 52 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> ________________________ </td>
   <td style="text-align:left;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Group variables </td>
   <td style="text-align:left;"> None </td>
  </tr>
</tbody>
</table>


**Variable type: factor**

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> skim_variable </th>
   <th style="text-align:right;"> n_missing </th>
   <th style="text-align:right;"> complete_rate </th>
   <th style="text-align:left;"> ordered </th>
   <th style="text-align:right;"> n_unique </th>
   <th style="text-align:left;"> top_counts </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> user_name </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:right;"> 6 </td>
   <td style="text-align:left;"> ade: 3892, cha: 3536, jer: 3402, car: 3112 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> classe </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:left;"> A: 5580, B: 3797, E: 3607, C: 3422 </td>
  </tr>
</tbody>
</table>


**Variable type: numeric**

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> skim_variable </th>
   <th style="text-align:right;"> n_missing </th>
   <th style="text-align:right;"> complete_rate </th>
   <th style="text-align:right;"> mean </th>
   <th style="text-align:right;"> sd </th>
   <th style="text-align:right;"> p0 </th>
   <th style="text-align:right;"> p25 </th>
   <th style="text-align:right;"> p50 </th>
   <th style="text-align:right;"> p75 </th>
   <th style="text-align:right;"> p100 </th>
   <th style="text-align:left;"> hist </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> roll_belt </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 64.41 </td>
   <td style="text-align:right;"> 62.75 </td>
   <td style="text-align:right;"> -28.90 </td>
   <td style="text-align:right;"> 1.10 </td>
   <td style="text-align:right;"> 113.00 </td>
   <td style="text-align:right;"> 123.00 </td>
   <td style="text-align:right;"> 162.00 </td>
   <td style="text-align:left;"> ▇▁▁▅▅ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> pitch_belt </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.31 </td>
   <td style="text-align:right;"> 22.35 </td>
   <td style="text-align:right;"> -55.80 </td>
   <td style="text-align:right;"> 1.76 </td>
   <td style="text-align:right;"> 5.28 </td>
   <td style="text-align:right;"> 14.90 </td>
   <td style="text-align:right;"> 60.30 </td>
   <td style="text-align:left;"> ▃▁▇▅▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> yaw_belt </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -11.21 </td>
   <td style="text-align:right;"> 95.19 </td>
   <td style="text-align:right;"> -180.00 </td>
   <td style="text-align:right;"> -88.30 </td>
   <td style="text-align:right;"> -13.00 </td>
   <td style="text-align:right;"> 12.90 </td>
   <td style="text-align:right;"> 179.00 </td>
   <td style="text-align:left;"> ▁▇▅▁▃ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> total_accel_belt </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 11.31 </td>
   <td style="text-align:right;"> 7.74 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 3.00 </td>
   <td style="text-align:right;"> 17.00 </td>
   <td style="text-align:right;"> 18.00 </td>
   <td style="text-align:right;"> 29.00 </td>
   <td style="text-align:left;"> ▇▁▂▆▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_belt_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -0.01 </td>
   <td style="text-align:right;"> 0.21 </td>
   <td style="text-align:right;"> -1.04 </td>
   <td style="text-align:right;"> -0.03 </td>
   <td style="text-align:right;"> 0.03 </td>
   <td style="text-align:right;"> 0.11 </td>
   <td style="text-align:right;"> 2.22 </td>
   <td style="text-align:left;"> ▁▇▁▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_belt_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.04 </td>
   <td style="text-align:right;"> 0.08 </td>
   <td style="text-align:right;"> -0.64 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 0.02 </td>
   <td style="text-align:right;"> 0.11 </td>
   <td style="text-align:right;"> 0.64 </td>
   <td style="text-align:left;"> ▁▁▇▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_belt_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -0.13 </td>
   <td style="text-align:right;"> 0.24 </td>
   <td style="text-align:right;"> -1.46 </td>
   <td style="text-align:right;"> -0.20 </td>
   <td style="text-align:right;"> -0.10 </td>
   <td style="text-align:right;"> -0.02 </td>
   <td style="text-align:right;"> 1.62 </td>
   <td style="text-align:left;"> ▁▂▇▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_belt_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -5.59 </td>
   <td style="text-align:right;"> 29.64 </td>
   <td style="text-align:right;"> -120.00 </td>
   <td style="text-align:right;"> -21.00 </td>
   <td style="text-align:right;"> -15.00 </td>
   <td style="text-align:right;"> -5.00 </td>
   <td style="text-align:right;"> 85.00 </td>
   <td style="text-align:left;"> ▁▁▇▁▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_belt_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 30.15 </td>
   <td style="text-align:right;"> 28.58 </td>
   <td style="text-align:right;"> -69.00 </td>
   <td style="text-align:right;"> 3.00 </td>
   <td style="text-align:right;"> 35.00 </td>
   <td style="text-align:right;"> 61.00 </td>
   <td style="text-align:right;"> 164.00 </td>
   <td style="text-align:left;"> ▁▇▇▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_belt_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -72.59 </td>
   <td style="text-align:right;"> 100.45 </td>
   <td style="text-align:right;"> -275.00 </td>
   <td style="text-align:right;"> -162.00 </td>
   <td style="text-align:right;"> -152.00 </td>
   <td style="text-align:right;"> 27.00 </td>
   <td style="text-align:right;"> 105.00 </td>
   <td style="text-align:left;"> ▁▇▁▅▃ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_belt_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 55.60 </td>
   <td style="text-align:right;"> 64.18 </td>
   <td style="text-align:right;"> -52.00 </td>
   <td style="text-align:right;"> 9.00 </td>
   <td style="text-align:right;"> 35.00 </td>
   <td style="text-align:right;"> 59.00 </td>
   <td style="text-align:right;"> 485.00 </td>
   <td style="text-align:left;"> ▇▁▂▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_belt_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 593.68 </td>
   <td style="text-align:right;"> 35.68 </td>
   <td style="text-align:right;"> 354.00 </td>
   <td style="text-align:right;"> 581.00 </td>
   <td style="text-align:right;"> 601.00 </td>
   <td style="text-align:right;"> 610.00 </td>
   <td style="text-align:right;"> 673.00 </td>
   <td style="text-align:left;"> ▁▁▁▇▃ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_belt_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -345.48 </td>
   <td style="text-align:right;"> 65.21 </td>
   <td style="text-align:right;"> -623.00 </td>
   <td style="text-align:right;"> -375.00 </td>
   <td style="text-align:right;"> -320.00 </td>
   <td style="text-align:right;"> -306.00 </td>
   <td style="text-align:right;"> 293.00 </td>
   <td style="text-align:left;"> ▁▇▁▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> roll_arm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 17.83 </td>
   <td style="text-align:right;"> 72.74 </td>
   <td style="text-align:right;"> -180.00 </td>
   <td style="text-align:right;"> -31.78 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 77.30 </td>
   <td style="text-align:right;"> 180.00 </td>
   <td style="text-align:left;"> ▁▃▇▆▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> pitch_arm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -4.61 </td>
   <td style="text-align:right;"> 30.68 </td>
   <td style="text-align:right;"> -88.80 </td>
   <td style="text-align:right;"> -25.90 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 11.20 </td>
   <td style="text-align:right;"> 88.50 </td>
   <td style="text-align:left;"> ▁▅▇▂▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> yaw_arm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -0.62 </td>
   <td style="text-align:right;"> 71.36 </td>
   <td style="text-align:right;"> -180.00 </td>
   <td style="text-align:right;"> -43.10 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 45.88 </td>
   <td style="text-align:right;"> 180.00 </td>
   <td style="text-align:left;"> ▁▃▇▃▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> total_accel_arm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 25.51 </td>
   <td style="text-align:right;"> 10.52 </td>
   <td style="text-align:right;"> 1.00 </td>
   <td style="text-align:right;"> 17.00 </td>
   <td style="text-align:right;"> 27.00 </td>
   <td style="text-align:right;"> 33.00 </td>
   <td style="text-align:right;"> 66.00 </td>
   <td style="text-align:left;"> ▃▆▇▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_arm_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.04 </td>
   <td style="text-align:right;"> 1.99 </td>
   <td style="text-align:right;"> -6.37 </td>
   <td style="text-align:right;"> -1.33 </td>
   <td style="text-align:right;"> 0.08 </td>
   <td style="text-align:right;"> 1.57 </td>
   <td style="text-align:right;"> 4.87 </td>
   <td style="text-align:left;"> ▁▃▇▆▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_arm_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -0.26 </td>
   <td style="text-align:right;"> 0.85 </td>
   <td style="text-align:right;"> -3.44 </td>
   <td style="text-align:right;"> -0.80 </td>
   <td style="text-align:right;"> -0.24 </td>
   <td style="text-align:right;"> 0.14 </td>
   <td style="text-align:right;"> 2.84 </td>
   <td style="text-align:left;"> ▁▂▇▂▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_arm_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.27 </td>
   <td style="text-align:right;"> 0.55 </td>
   <td style="text-align:right;"> -2.33 </td>
   <td style="text-align:right;"> -0.07 </td>
   <td style="text-align:right;"> 0.23 </td>
   <td style="text-align:right;"> 0.72 </td>
   <td style="text-align:right;"> 3.02 </td>
   <td style="text-align:left;"> ▁▂▇▂▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_arm_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -60.24 </td>
   <td style="text-align:right;"> 182.04 </td>
   <td style="text-align:right;"> -404.00 </td>
   <td style="text-align:right;"> -242.00 </td>
   <td style="text-align:right;"> -44.00 </td>
   <td style="text-align:right;"> 84.00 </td>
   <td style="text-align:right;"> 437.00 </td>
   <td style="text-align:left;"> ▇▅▇▅▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_arm_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 32.60 </td>
   <td style="text-align:right;"> 109.87 </td>
   <td style="text-align:right;"> -318.00 </td>
   <td style="text-align:right;"> -54.00 </td>
   <td style="text-align:right;"> 14.00 </td>
   <td style="text-align:right;"> 139.00 </td>
   <td style="text-align:right;"> 308.00 </td>
   <td style="text-align:left;"> ▁▃▇▆▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_arm_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -71.25 </td>
   <td style="text-align:right;"> 134.65 </td>
   <td style="text-align:right;"> -636.00 </td>
   <td style="text-align:right;"> -143.00 </td>
   <td style="text-align:right;"> -47.00 </td>
   <td style="text-align:right;"> 23.00 </td>
   <td style="text-align:right;"> 292.00 </td>
   <td style="text-align:left;"> ▁▁▅▇▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_arm_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 191.72 </td>
   <td style="text-align:right;"> 443.64 </td>
   <td style="text-align:right;"> -584.00 </td>
   <td style="text-align:right;"> -300.00 </td>
   <td style="text-align:right;"> 289.00 </td>
   <td style="text-align:right;"> 637.00 </td>
   <td style="text-align:right;"> 782.00 </td>
   <td style="text-align:left;"> ▆▃▂▃▇ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_arm_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 156.61 </td>
   <td style="text-align:right;"> 201.91 </td>
   <td style="text-align:right;"> -392.00 </td>
   <td style="text-align:right;"> -9.00 </td>
   <td style="text-align:right;"> 202.00 </td>
   <td style="text-align:right;"> 323.00 </td>
   <td style="text-align:right;"> 583.00 </td>
   <td style="text-align:left;"> ▁▅▅▇▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_arm_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 306.49 </td>
   <td style="text-align:right;"> 326.62 </td>
   <td style="text-align:right;"> -597.00 </td>
   <td style="text-align:right;"> 131.25 </td>
   <td style="text-align:right;"> 444.00 </td>
   <td style="text-align:right;"> 545.00 </td>
   <td style="text-align:right;"> 694.00 </td>
   <td style="text-align:left;"> ▁▂▂▃▇ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> roll_dumbbell </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 23.84 </td>
   <td style="text-align:right;"> 69.93 </td>
   <td style="text-align:right;"> -153.71 </td>
   <td style="text-align:right;"> -18.49 </td>
   <td style="text-align:right;"> 48.17 </td>
   <td style="text-align:right;"> 67.61 </td>
   <td style="text-align:right;"> 153.55 </td>
   <td style="text-align:left;"> ▂▂▃▇▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> pitch_dumbbell </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -10.78 </td>
   <td style="text-align:right;"> 36.99 </td>
   <td style="text-align:right;"> -149.59 </td>
   <td style="text-align:right;"> -40.89 </td>
   <td style="text-align:right;"> -20.96 </td>
   <td style="text-align:right;"> 17.50 </td>
   <td style="text-align:right;"> 149.40 </td>
   <td style="text-align:left;"> ▁▆▇▂▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> yaw_dumbbell </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1.67 </td>
   <td style="text-align:right;"> 82.52 </td>
   <td style="text-align:right;"> -150.87 </td>
   <td style="text-align:right;"> -77.64 </td>
   <td style="text-align:right;"> -3.32 </td>
   <td style="text-align:right;"> 79.64 </td>
   <td style="text-align:right;"> 154.95 </td>
   <td style="text-align:left;"> ▃▇▅▅▆ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> total_accel_dumbbell </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 13.72 </td>
   <td style="text-align:right;"> 10.23 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 4.00 </td>
   <td style="text-align:right;"> 10.00 </td>
   <td style="text-align:right;"> 19.00 </td>
   <td style="text-align:right;"> 58.00 </td>
   <td style="text-align:left;"> ▇▅▃▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_dumbbell_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.16 </td>
   <td style="text-align:right;"> 1.51 </td>
   <td style="text-align:right;"> -204.00 </td>
   <td style="text-align:right;"> -0.03 </td>
   <td style="text-align:right;"> 0.13 </td>
   <td style="text-align:right;"> 0.35 </td>
   <td style="text-align:right;"> 2.22 </td>
   <td style="text-align:left;"> ▁▁▁▁▇ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_dumbbell_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0.61 </td>
   <td style="text-align:right;"> -2.10 </td>
   <td style="text-align:right;"> -0.14 </td>
   <td style="text-align:right;"> 0.03 </td>
   <td style="text-align:right;"> 0.21 </td>
   <td style="text-align:right;"> 52.00 </td>
   <td style="text-align:left;"> ▇▁▁▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_dumbbell_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -0.13 </td>
   <td style="text-align:right;"> 2.29 </td>
   <td style="text-align:right;"> -2.38 </td>
   <td style="text-align:right;"> -0.31 </td>
   <td style="text-align:right;"> -0.13 </td>
   <td style="text-align:right;"> 0.03 </td>
   <td style="text-align:right;"> 317.00 </td>
   <td style="text-align:left;"> ▇▁▁▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_dumbbell_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -28.62 </td>
   <td style="text-align:right;"> 67.32 </td>
   <td style="text-align:right;"> -419.00 </td>
   <td style="text-align:right;"> -50.00 </td>
   <td style="text-align:right;"> -8.00 </td>
   <td style="text-align:right;"> 11.00 </td>
   <td style="text-align:right;"> 235.00 </td>
   <td style="text-align:left;"> ▁▁▆▇▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_dumbbell_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 52.63 </td>
   <td style="text-align:right;"> 80.75 </td>
   <td style="text-align:right;"> -189.00 </td>
   <td style="text-align:right;"> -8.00 </td>
   <td style="text-align:right;"> 41.50 </td>
   <td style="text-align:right;"> 111.00 </td>
   <td style="text-align:right;"> 315.00 </td>
   <td style="text-align:left;"> ▁▇▇▅▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_dumbbell_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -38.32 </td>
   <td style="text-align:right;"> 109.47 </td>
   <td style="text-align:right;"> -334.00 </td>
   <td style="text-align:right;"> -142.00 </td>
   <td style="text-align:right;"> -1.00 </td>
   <td style="text-align:right;"> 38.00 </td>
   <td style="text-align:right;"> 318.00 </td>
   <td style="text-align:left;"> ▁▆▇▃▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_dumbbell_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -328.48 </td>
   <td style="text-align:right;"> 339.72 </td>
   <td style="text-align:right;"> -643.00 </td>
   <td style="text-align:right;"> -535.00 </td>
   <td style="text-align:right;"> -479.00 </td>
   <td style="text-align:right;"> -304.00 </td>
   <td style="text-align:right;"> 592.00 </td>
   <td style="text-align:left;"> ▇▂▁▁▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_dumbbell_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 220.97 </td>
   <td style="text-align:right;"> 326.87 </td>
   <td style="text-align:right;"> -3600.00 </td>
   <td style="text-align:right;"> 231.00 </td>
   <td style="text-align:right;"> 311.00 </td>
   <td style="text-align:right;"> 390.00 </td>
   <td style="text-align:right;"> 633.00 </td>
   <td style="text-align:left;"> ▁▁▁▁▇ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_dumbbell_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 46.05 </td>
   <td style="text-align:right;"> 139.96 </td>
   <td style="text-align:right;"> -262.00 </td>
   <td style="text-align:right;"> -45.00 </td>
   <td style="text-align:right;"> 13.00 </td>
   <td style="text-align:right;"> 95.00 </td>
   <td style="text-align:right;"> 452.00 </td>
   <td style="text-align:left;"> ▁▇▆▂▂ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> roll_forearm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 33.83 </td>
   <td style="text-align:right;"> 108.04 </td>
   <td style="text-align:right;"> -180.00 </td>
   <td style="text-align:right;"> -0.74 </td>
   <td style="text-align:right;"> 21.70 </td>
   <td style="text-align:right;"> 140.00 </td>
   <td style="text-align:right;"> 180.00 </td>
   <td style="text-align:left;"> ▃▂▇▂▇ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> pitch_forearm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 10.71 </td>
   <td style="text-align:right;"> 28.15 </td>
   <td style="text-align:right;"> -72.50 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 9.24 </td>
   <td style="text-align:right;"> 28.40 </td>
   <td style="text-align:right;"> 89.80 </td>
   <td style="text-align:left;"> ▁▁▇▃▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> yaw_forearm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 19.21 </td>
   <td style="text-align:right;"> 103.22 </td>
   <td style="text-align:right;"> -180.00 </td>
   <td style="text-align:right;"> -68.60 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 110.00 </td>
   <td style="text-align:right;"> 180.00 </td>
   <td style="text-align:left;"> ▅▅▇▆▇ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> total_accel_forearm </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 34.72 </td>
   <td style="text-align:right;"> 10.06 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 29.00 </td>
   <td style="text-align:right;"> 36.00 </td>
   <td style="text-align:right;"> 41.00 </td>
   <td style="text-align:right;"> 108.00 </td>
   <td style="text-align:left;"> ▁▇▂▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_forearm_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.16 </td>
   <td style="text-align:right;"> 0.65 </td>
   <td style="text-align:right;"> -22.00 </td>
   <td style="text-align:right;"> -0.22 </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0.56 </td>
   <td style="text-align:right;"> 3.97 </td>
   <td style="text-align:left;"> ▁▁▁▁▇ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_forearm_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.08 </td>
   <td style="text-align:right;"> 3.10 </td>
   <td style="text-align:right;"> -7.02 </td>
   <td style="text-align:right;"> -1.46 </td>
   <td style="text-align:right;"> 0.03 </td>
   <td style="text-align:right;"> 1.62 </td>
   <td style="text-align:right;"> 311.00 </td>
   <td style="text-align:left;"> ▇▁▁▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> gyros_forearm_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0.15 </td>
   <td style="text-align:right;"> 1.75 </td>
   <td style="text-align:right;"> -8.09 </td>
   <td style="text-align:right;"> -0.18 </td>
   <td style="text-align:right;"> 0.08 </td>
   <td style="text-align:right;"> 0.49 </td>
   <td style="text-align:right;"> 231.00 </td>
   <td style="text-align:left;"> ▇▁▁▁▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_forearm_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -61.65 </td>
   <td style="text-align:right;"> 180.59 </td>
   <td style="text-align:right;"> -498.00 </td>
   <td style="text-align:right;"> -178.00 </td>
   <td style="text-align:right;"> -57.00 </td>
   <td style="text-align:right;"> 76.00 </td>
   <td style="text-align:right;"> 477.00 </td>
   <td style="text-align:left;"> ▂▆▇▅▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_forearm_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 163.66 </td>
   <td style="text-align:right;"> 200.13 </td>
   <td style="text-align:right;"> -632.00 </td>
   <td style="text-align:right;"> 57.00 </td>
   <td style="text-align:right;"> 201.00 </td>
   <td style="text-align:right;"> 312.00 </td>
   <td style="text-align:right;"> 923.00 </td>
   <td style="text-align:left;"> ▁▂▇▅▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> accel_forearm_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -55.29 </td>
   <td style="text-align:right;"> 138.40 </td>
   <td style="text-align:right;"> -446.00 </td>
   <td style="text-align:right;"> -182.00 </td>
   <td style="text-align:right;"> -39.00 </td>
   <td style="text-align:right;"> 26.00 </td>
   <td style="text-align:right;"> 291.00 </td>
   <td style="text-align:left;"> ▁▇▅▅▃ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_forearm_x </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> -312.58 </td>
   <td style="text-align:right;"> 346.96 </td>
   <td style="text-align:right;"> -1280.00 </td>
   <td style="text-align:right;"> -616.00 </td>
   <td style="text-align:right;"> -378.00 </td>
   <td style="text-align:right;"> -73.00 </td>
   <td style="text-align:right;"> 672.00 </td>
   <td style="text-align:left;"> ▁▇▇▅▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_forearm_y </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 380.12 </td>
   <td style="text-align:right;"> 509.37 </td>
   <td style="text-align:right;"> -896.00 </td>
   <td style="text-align:right;"> 2.00 </td>
   <td style="text-align:right;"> 591.00 </td>
   <td style="text-align:right;"> 737.00 </td>
   <td style="text-align:right;"> 1480.00 </td>
   <td style="text-align:left;"> ▂▂▂▇▁ </td>
  </tr>
  <tr>
   <td style="text-align:left;"> magnet_forearm_z </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 393.61 </td>
   <td style="text-align:right;"> 369.27 </td>
   <td style="text-align:right;"> -973.00 </td>
   <td style="text-align:right;"> 191.00 </td>
   <td style="text-align:right;"> 511.00 </td>
   <td style="text-align:right;"> 653.00 </td>
   <td style="text-align:right;"> 1090.00 </td>
   <td style="text-align:left;"> ▁▁▂▇▃ </td>
  </tr>
</tbody>
</table>

The resultant data set only includes complete cases.  Examination of the 
numeric variable summary statistics and thumbnail histograms indicate many of 
the features display non-Gaussian indications.  The analysis will use variable 
transformations to standardize the data.  Due to the presence of negative and 
zero values, a Yeo-Johnson transformation is utilized.    

For this analysis, the *training* data is partitioned into *training* 
and *validation* subsets.  The *testing* data does not contain the *classe* 
variable and is dedicated to a prediction test of 20 observations.

A Principal Component Analysis variance-explained plot of the *training* set 
indicates PCA transformation could reduce the number of features and thus 
improve computational efficiency.  The variance explained per principal 
component asymptotically reaches a limit around the 20th principal component.       


```r
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
training <- training[inTrain, ]
validation <- training[-inTrain, ]

trainingPCA <- prcomp(x = training[-c(1,54)], scale = F, center = T)
pcaVar <- trainingPCA$sdev^2
varExp <- pcaVar / sum(pcaVar)
plot(varExp, xlab = "Principal Component", ylab = "Proportion of Variance Explained", 
	ylim = c(0,.3), type = "b")
```

<img src="figure/EDA2-1.png" title="plot of chunk EDA2" alt="plot of chunk EDA2" style="display: block; margin: auto;" />

### Modeling

This analysis uses an ensemble of three different machine learning algorithms 
stacked via a simple majority voting scheme.  If all three models disagree, the 
model with the highest accuracy on the *validation* set provides the answer. 
As discussed above, *Yeo-Johnson* and *pca* transformations are used to 
standardize and scale the data. *Center* and *scale* transformations are 
prerequisites for *pca* transformation, and a *near-zero variance* transformation 
is used to check for isolated features that might bias the predictions.

The *caret* calculated PCA analysis indicates slightly more than the preliminary 
estimate of principle components are required to capture 95% of the variance. 
Additionally, the lack of additional variable removal during *caret* preprocessing 
indicates no features possess near-zero variance.  


```r
trainingPP <- preProcess(training[, -54], method = c("center", "scale", "YeoJohnson", "nzv", "pca"))
training1 <- predict(trainingPP, newdata = training)
validation1 <- predict(trainingPP, newdata = validation)
trainingPP  # No NSV features; passed 53, ignored user, transformed 
```

```
## Created from 13737 samples and 53 variables
## 
## Pre-processing:
##   - centered (52)
##   - ignored (1)
##   - principal component signal extraction (52)
##   - scaled (52)
##   - Yeo-Johnson transformation (51)
## 
## Lambda estimates for Yeo-Johnson transformation:
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##  0.2606  0.9079  0.9976  0.9744  1.0673  1.4373 
## 
## PCA needed 28 components to capture 95 percent of the variance
```

To improve accuracy, all three algorithms in the stacked ensemble utilize some form of boosting or 
bagging.  All were chosen for their predictive abilities with classification 
problems.  The three algorithms in the ensemble are:
* *Random Forest* via *ranger*
* *Stochastic Gradient Boosting* via *gbm*
* *Bagged Classification and Regression Tree* via *treebag*

To prevent overfitting, k-fold cross-validation using 10 folds and five repetitions 
is utilized.  These numbers were chosen as a compromise between modeling desires 
and required computation time.  For computational efficiency, parallel 
computation via a multi-core processor and the *doParallel* package is employed. 


```r
set.seed(12345)
cl <- makePSOCKcluster(3)  # doParallel package for parallel processing
registerDoParallel(cl)  # doParallel package for parallel processing

modControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
gbmFit <- train(classe ~ ., data = training1, method = "gbm", trControl = modControl, verbose = FALSE)
rfFit <- train(classe ~ ., data = training1, method = "ranger", trControl = modControl, verbose = FALSE)
treefit <- train(classe ~ ., data = training1, method = "treebag", trControl = modControl, verbose = FALSE)

stopCluster(cl)  # doParallel package for parallel processing

gbmVote <- predict(gbmFit, newdata = validation1)
rfVote <- predict(rfFit, newdata = validation1)
treeVote <- predict(treefit, newdata = validation1)
voting <- function(a, b, c) {
  for(i in 1:length(a)) {
    if(b[i] == c[i]) a[i] = b[i]
  }
  return(a)
}
voteTally <- voting(rfVote, gbmVote, treeVote)
```

The random forest's out-of-bag prediction error estimate based on the training 
set is provided below.  


```r
rfFit$finalModel
```

```
## Ranger result
## 
## Call:
##  ranger::ranger(dependent.variable.name = ".outcome", data = x,      mtry = min(param$mtry, ncol(x)), min.node.size = param$min.node.size,      splitrule = as.character(param$splitrule), write.forest = TRUE,      probability = classProbs, ...) 
## 
## Type:                             Classification 
## Number of trees:                  500 
## Sample size:                      13737 
## Number of independent variables:  33 
## Mtry:                             33 
## Target node size:                 1 
## Variable importance mode:         none 
## Splitrule:                        extratrees 
## Number of random splits:          1 
## OOB prediction error:             1.89 %
```

The three model's accuracies on the validation set are provided below. The 
random forest model perfectly predicted the validation set, exceeding the ~98% 
OOB training set-derived estimate, and the bagged tree model only missed 
perfection by a few elements.  The boosted gradient tree performed slightly 
worse at ~86% accuracy.


```r
postResample(pred = rfVote, obs = validation1$classe)
```

```
## Accuracy    Kappa 
##        1        1
```

```r
postResample(pred = treeVote, obs = validation1$classe)
```

```
##  Accuracy     Kappa 
## 0.9997577 0.9996933
```

```r
postResample(pred = gbmVote, obs = validation1$classe)
```

```
##  Accuracy     Kappa 
## 0.8589775 0.8212099
```

The final ensemble confusion matrix and accuracies on the validation set are 
provided below.  In one case, the less accurate models outvoted the random 
forest model to the minute detriment of overall accuracy.  Because of the tiny 
difference and the upside potential of the ensemble against future unknown data, 
this project retains the ensemble approach.


```r
confusionMatrix(voteTally, validation1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1186    0    0    0    0
##          B    0  798    0    0    0
##          C    0    0  709    0    1
##          D    0    0    0  679    0
##          E    0    0    0    0  754
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9998     
##                  95% CI : (0.9987, 1)
##     No Information Rate : 0.2874     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9997     
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   0.9987
## Specificity            1.0000   1.0000   0.9997   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   0.9986   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   0.9997
## Prevalence             0.2874   0.1934   0.1718   0.1645   0.1829
## Detection Rate         0.2874   0.1934   0.1718   0.1645   0.1827
## Detection Prevalence   0.2874   0.1934   0.1720   0.1645   0.1827
## Balanced Accuracy      1.0000   1.0000   0.9999   1.0000   0.9993
```

### Predictions

The predictions of the testing set variables are provided below.
 

```r
testing <- testing %>%
  select(where(~mean(is.na(.))< 0.9)) %>%
  select(-c(X1, raw_timestamp_part_1, raw_timestamp_part_2,
            cvtd_timestamp, new_window, num_window)) %>%
  mutate(user_name = factor(user_name))
testing1 <- predict(trainingPP, newdata = testing)
predict(rfFit, newdata = testing1)
```

```
##  [1] B A C A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Conclusions

The simple majority stacked ensemble utilized predicts the validation set with 
99.98% accuracy. The individual algorithms performance on the *validation* set 
indicate the ensemble approach is probably unnecessary in this case, but that 
might not be the case with other datasets.    
