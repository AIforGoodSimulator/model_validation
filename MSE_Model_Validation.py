#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------
Implementing Mean Squared Error Using sklearn
Source: https://www.geeksforgeeks.org/python-mean-squared-error/
-------------------------------------------------------
-------------------------------------------------------
author:  Chandra Manivannan

version date: 11 September 2020

-------------------------------------------------------
"""


from sklearn.metrics import mean_squared_error 
  
# Given values 
Y_true = [1,1,2,2,4]  # Y_true = Y (original values) 
  
# calculated values 
Y_pred = [0.6,1.29,1.99,2.69,3.4]  # Y_pred = Y' 
  
# Calculation of Mean Squared Error (MSE) 
mean_squared_error(Y_true,Y_pred) 

