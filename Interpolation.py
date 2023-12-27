import numpy as np


# bilinear interpolation
# input is the sourrounding 4 points and location of P
# output is the value of P
def bilinear_intep(Q1,Q2,Q3,Q4,P):
    
    upper_Q12_delt = Q2[0] - Q1[0]
    upper_Q1_P_delt = P[0] - Q1[0]
    upper_Q2_P_delt = Q2[0] - P[0] 
    upper_var = upper_Q1_P_delt / upper_Q12_delt * Q2[2] + upper_Q2_P_delt / upper_Q12_delt * Q1[2]

    lower_Q43_delt = Q3[0] - Q4[0]
    lower_Q3_p_delt = Q3[0] - P[0]
    lower_Q4_p_delt = P[0] - Q4[0]
    lower_var = lower_Q4_p_delt / lower_Q43_delt *Q3[2] + lower_Q3_p_delt / lower_Q43_delt * Q4[2]

    mid_Q12_delt = Q1[1] - Q3[1]
    mid_Q1_p_delt = Q1[1] -  P[1]
    mid_Q3_p_delt = P[1] - Q3[1]
    mid_var = mid_Q1_p_delt / mid_Q12_delt *lower_var +  mid_Q3_p_delt / mid_Q12_delt *upper_var

    return mid_var



