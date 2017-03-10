# -*- coding: utf-8 -*-

import csv
import math

def func_read_csv(file, *args):
    result1 = []
    result2 = []

    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            result1.append(float(row[args[0]]))
            result2.append(float(row[args[1]]))
    # print(result1,result2)
    return (result1, result2)

def func_generate_d(list1, list2):
    result = []
    length1 = len(list1)
    length2 = len(list2)
    length = length1  if length1 <= length2 else length2
    for i in range(length):
        result.append(list2[i] - list1[i])
    return result

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
    min1 = min(list_para)
    max1 = max(list_para)
    mean1 = calc_mean(list_para)
    var1 = calc_variance(list_para, mean1)
    sd1 = calc_standard_variance(var1)
    return [min1, max1, mean1, var1, sd1]

def func_print_result(list_para):
    print("*" * 20)
    print("Print statistics")
    print("Min is :", list_para[0])
    print("Max is :", list_para[1])
    print("Mean is :", list_para[2])
    print("Variance is :", list_para[3])
    print("Standard Deviation is :", list_para[4])
    print("*" * 20)
    print(" ")

def main():
    result1, result2 = func_read_csv("stroopdata.csv", "Congruent", "Incongruent")
    result3 = func_generate_d(result1, result2)
    func_print_result(calc_statistics(result1))
    func_print_result(calc_statistics(result2))
    func_print_result(calc_statistics(result3))

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