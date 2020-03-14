import os
import numpy as np
import math
def convert_fixed_pt_round(t,t_precision):
    if t_precision != None:
        t = 1.0*round(t*pow(2,t_precision),0)/pow(2,t_precision)
    return t

def convert_fixed_pt(t,t_precision):
    if t_precision != None:
        #t = 1.0*round(t*pow(2,t_precision),0)/pow(2,t_precision)
        t = 1.0*math.floor(t*pow(2,t_precision))/pow(2,t_precision)
    return t


z = (convert_fixed_pt(125, 2)) 
print (z)


# Python program to convert decimal to binary 
    
def decimalToBinary(z):
  if z > 1:
    decimalToBinary(z // 2)
  print(z % 2, end = '')
number = int(z)

#main() function
decimalToBinary(number)
    
