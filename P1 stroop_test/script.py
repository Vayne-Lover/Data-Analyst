# -*- coding: utf-8 -*-

import csv
import math

def func_read_csv(file,*args):
    result1 = []
    result2 = []

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result1.append(float(row["Congruent"]))
            result2.append(float(row["Incongruent"]))
    # print(result1,result2)
    return (result1, result2)

def calc_mean(list_para):
    result = 0.0
    length = len(list_para)
    for i in list_para:
        result += i
    return result / length

def calc_variance(list_para, mean_para):
    result = 0.0
    for i in list_para:
        result += (i - mean_para) ** 2
    length = len(list_para) - 1
    if length <= 0:
        length = 1
    return result / length

def calc_standard_variance(var_para):
    return math.sqrt(var_para)

def calc_statistics(list_para):
    mean1 = calc_mean(list_para)
    var1 = calc_variance(list_para, mean1)
    sd1 = calc_standard_variance(var1)
    return [mean1, var1, sd1]

def func_print_result(list_para):
    print("*" * 10)
    print("Print statistics")
    print("Mean is :", list_para[0])
    print("Variance is :", list_para[1])
    print("SD is :", list_para[2])
    print("*" * 10)

def main():
    result1, result2 = func_read_csv("stroopdata.csv", "Congruent", "Incongruent")
    func_print_result(calc_statistics(result1))
    func_print_result(calc_statistics(result2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error")
        print(e)
    else:
        print("Success!")
    finally:
        print("Finish!")