import random
import time
import math
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt

def CreatePlot(input_data, exec_time, algo_name):
    plt.xlabel('Ulaz [n]')
    plt.ylabel('Vreme [ms]')
    plt.plot(input_data, exec_time, '-', label = algo_name)
    plt.legend()
    print(algo_name)
    for i in range(0, len(input_data)):
        print("input_data: ", input_data[i], ", exec_time: ", exec_time[i])


def RandomList(min, max, elements):
    list = random.sample(range(min, max), elements)
    return list


def SelectionSort(Array):
    for i in range(len(Array) - 1):
        minEl = i
        for j in range(i + 1, len(Array)):
            if(Array[j] < Array[minEl]):
                minEl = j
    Array[minEl], Array[i] = Array[i], Array[minEl]

def SortedTest(Array):
    testList = Array[:]
    testList.sort()
    if testList == Array:
        return True
    return False


def TestAndPlotAlgorithm(Algorithm):
    algo_name = Algorithm.__name__
    test_ranges = [10,100,1000,10000]
    input_data = []
    exec_time = []
    for n in test_ranges:
        Array = RandomList(0, 1000000 + 1, n)
        start_time = time.clock()
        Algorithm(Array)
        end_time = time.clock()
        exec_time.append((end_time - start_time))
        input_data.append(n)
    if(SortedTest):
        CreatePlot(input_data, exec_time, algo_name)

def Parent(i):
    return (i - 1) // 2


def Left(i):
    return 2 * i + 1


def Right(i):
    return 2 * i + 2

def MaxHeapify(Array, i):
    l = Left(i)
    r = Right(i)
    if l < heap_size and Array[l] > Array[i]:
        largest = l
    else:
        largest = i
    if r < heap_size and Array[r] > Array[largest]:
        largest = r
    if largest != i:
        Array[i], Array[largest] = Array[largest], Array[i]
        MaxHeapify(Array, largest)


def BuildMaxHeap(Array):
    global heap_size
    heap_size = len(Array)
    itterate = len(Array) // 2 - 1
    for i in range(itterate, -1, -1):
        MaxHeapify(Array, i)


def HeapSort(Array):
    global heap_size
    BuildMaxHeap(Array)
    for i in range(len(Array) - 1, 0, -1):
        Array[0], Array[i] = Array[i], Array[0]
        heap_size -= 1
    MaxHeapify(Array, 0)


def countingSort(arr, exp1):
    n = len(arr)
    # The output array elements that will have sorted arr
    output = [0] * (n)
    # initialize count array as 0
    count = [0] * (10)
    # Store count of occurrences in count[]
    for i in range(0, n):
        index = (arr[i]/exp1)
        count[ int((index)%10) ] += 1
    # Change count[i] so that count[i] now contains actual
    #  position of this digit in output array
    for i in range(1,10):
        count[i] += count[i-1]
 
    # Build the output array
    i = n-1
    while i>=0:
        index = (arr[i]/exp1)
        output[ count[ int((index)%10) ] - 1] = arr[i]
        count[ int((index)%10) ] -= 1
        i -= 1
 
    # Copying the output array to arr[],
    # so that arr now contains sorted numbers
    i = 0
    for i in range(0,len(arr)):
        arr[i] = output[i]
 
# Method to do Radix Sort
def radixSort(arr):
 
    # Find the maximum number to know number of digits
    max1 = max(arr)
 
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1/exp > 0:
        countingSort(arr,exp)
        exp *= 10

TestAndPlotAlgorithm(SelectionSort)
TestAndPlotAlgorithm(HeapSort)
TestAndPlotAlgorithm(radixSort)
plt.show()
plt.show()
