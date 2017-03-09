# -*- coding: utf-8 -*-

import csv
import math

def func_read_csv(file,*args):
    result1=[]
    result2=[]

    with open(file) as csvfile:
        reader=csv.DictReader(csvfile)
        for row in reader:
            result1.append(float(row["Congruent"]))
            result2.append(float(row["Incongruent"]))
    print(result1,result2)
    return (result1,result2)

def calc_average(list_para):
    result=0.0
    length=len(list_para)
    for i in list_para:
        result += i
    return result/length

def calc_variance(list_para, avg_para):
    result=0.0
    for i in list_para:
        result+=(i-avg_para)**2
    length=len(list_para)-1
    return result/length

def calc_standard_variance(var_para):
    return math.sqrt(var_para)

def main():
    result1,result2 = func_read_csv("stroopdata.csv", "Congruent", "Incongruent")
    avg1=calc_average(result1)
    var1=calc_variance(result1,avg1)
    sd1=calc_standard_variance(var1)
    print(avg1,var1,sd1)

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(e)
        print("Error")
    else:
        print("Success!")
    finally:
        print("Finish!")