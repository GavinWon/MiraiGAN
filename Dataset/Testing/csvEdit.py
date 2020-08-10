# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:33:19 2020

@author: Gavin
"""

import csv

addType()

def addType():
    feature_text = "Type"
    default_text = 'udpplain'
    # Open the input_file in read mode and output_file in write mode
    with open('All CSV\\udpplain.csv', 'r') as read_obj, \
        open('All CSV\\udpplain_new.csv', 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = csv.reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = csv.writer(write_obj)
        
        firstRow = next(csv_reader)
        firstRow.append(feature_text)
        csv_writer.writerow(firstRow)
        
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Append the default text in the row / list
            row.append(default_text)
            # Add the updated row / list to the output file
            csv_writer.writerow(row)

def combineDataset():
    list = []
with open("all_data2.csv", "r") as readfile:
    reader = csv.reader(readfile)
    next(reader)
    previousRow = next(reader)
    list.append(previousRow)
    for row in reader:
        if row[1:] != previousRow[1:] and notCorrupted(row) and moreThan1000(row) and moreThan100Files(row):
            list.append(row)
        previousRow = row

dict = checkFilesPerFamily()

print(dict)

findStart()
        
        
        
with open('all_data2_new.csv', 'w', newline = '') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(list)
    
def createLabels():
    
