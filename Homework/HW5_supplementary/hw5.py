import numpy as np
import sys
import csv
import pandas as pd 
import matplotlib.pyplot as plt



def visualize_data(df):
    plt.plot(df["year"], df["days"])
    plt.title("Day Lake Frozen per year")
    plt.xlabel("Year")
    plt.ylabel("Days")
    plt.savefig("plot.jpg")

# Question 3: Linear Regression  
def linear_regression(df):
    #Q3a
    n = len(df)
    one = []
    for i in range(n):
        one.append(1)
    x_i = df["year"]
    X = np.column_stack([one,x_i])
    return X
    
def y_vector(df):
    #Q3b
    Y = np.array(df["days"])
    return Y 

def matrix_prod(df):
    #Q3c
    X = linear_regression(df)
    Xt = np.transpose(linear_regression(df))
    Z = np.dot(Xt,X)
    return Z

def inverse(df):
    #Q3d
    I = np.linalg.inv(matrix_prod(df))
    return I

def P_i(df):
    #Q3e
    Xt = np.matrix.transpose(linear_regression(df))
    I = inverse(df)
    PI = np.dot(I, Xt)
    return PI

def B_hat(df):
    #Q3f
    PI = P_i(df)
    Y = y_vector(df)
    b_hat = np.dot(PI, Y)
    return b_hat

# Question 4: Prediction
def predic(df):
    b_hat = B_hat(df)
    y_test = b_hat[0] + b_hat[1]*2022
    return y_test


# Question 5: Model Interpretation
def mod_int(df):
    b_hat = B_hat(df)
    if b_hat[1] < 0:
        return "<"

# Question 6: Model Limitation
def mod_lim(df):
    b_hat = B_hat(df)
    x = -b_hat[0]/b_hat[1]
    return x


if __name__ == "__main__":
    filepath = sys.argv[1]
    df = pd.read_csv(filepath)


    print ("Q2: ") 
    visualize_data(df)
    
    print ("Q3a: ") 
    print(linear_regression(df))

    print("Q3b: ")
    print(y_vector(df))

    print("Q3c: ") 
    print(matrix_prod(df))

    print("Q3d: ")
    print(inverse(df))


    print("Q3e: ")
    print(P_i(df))

    print("Q3f: ")
    print(B_hat(df))

    print("Q4: " + str(predic(df)))

    print("Q5a: ", mod_int(df))
    print("Q5b: Because Beta 1 is negative, it means that the number of days the lake is frozen is decreasing each year")

    print("Q6a: " +  str(mod_lim(df)))
    print("Q6b: This is a compelling prediction because we know that as the years increase the amount of ice decreases, so eventually there will be no more ice cover") 

    
