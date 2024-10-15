import sys
import math
import string
import math

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    
    alphabet = string.ascii_uppercase
    X=dict()
    for letter in alphabet:
        X[letter] = 0

    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        for line in f: 
            row = line.upper().strip().split(" ")
            for word in row: 
                for letter in word:
                    if letter not in X:
                        continue
                    else: 
                        X[letter] += 1                
    return X



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!


def q_2(filename, letter_position):
    pos_e = get_parameter_vectors()[0][letter_position]
    pos_s = get_parameter_vectors()[1][letter_position]
    letter_coefficient = list(shred(filename).values())[letter_position]
    
    return letter_coefficient * math.log(pos_e), letter_coefficient * math.log(pos_s)


def q_3(filename): 
    p_eng = 0.6
    p_spanish = 0.4
    
    sum_q2_e = 0
    sum_q2_s = 0
    
    for i in range(0,26):
        sum_q2_e += q_2(filename, i)[0] 
        sum_q2_s += q_2(filename, i)[1]
        
    F_eng = math.log(p_eng) + sum_q2_e
    F_span = math.log(p_spanish) + sum_q2_s
    
    return F_eng, F_span

def q_4(filename): 
    p = 0
    if q_3(filename)[1] - q_3(filename)[0] >= 100:
        p = 0.0000
    elif q_3(filename)[1] - q_3(filename)[0] <= -100:
        p = 1.0000
    else: 
        p = 1 / (1 + math.exp(q_3(filename)[1] - q_3(filename)[0]))
    return p



if __name__ == "__main__":
    print("Q1")
    for i in shred("letter.txt"):
        print("{i} {x}".format(i=i, x=shred("letter.txt").get(i))) 
        
    print("Q2")
    print("{e:0.4f}\n{s:0.4f}".format(e=q_2("letter.txt", 0)[0],s=q_2("letter.txt", 0)[1]) )

    print("Q3")
    print("{e:0.4f}\n{s:0.4f}".format(e=q_3("letter.txt")[0], s=q_3("letter.txt")[1]) )

    print("Q4")
    print("{e:0.4f}".format( e=q_4("letter.txt") ) )

