# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:33:19 2020

@author: Gavin
"""

import csv
import pandas as pd

addType()
combineDataset()
createLabels()

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
    all_filenames = ["ack_new.csv", "scan_new.csv", "syn_new.csv", "udp_new.csv", "udpplain_new.csv", "benign_traffic_new.csv"]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv.to_csv( "mirai_testing.csv", index=False, encoding='utf-8-sig')
    # list = []
    # for name in ["ack", "benign_traffic", "scan", "syn", "udp", "udpplain"]:
    #     filename = "All CSV\\" + name + "_new.csv"
    #     with open(filename, "r") as readfile:
    #         reader = csv.reader(readfile)
    #         next(reader)
    #         for row in reader:
    #             list.append(row)     
    #     with open('mirai_testing.csv', 'w', newline = '') as writeFile:
    #         writer = csv.writer(writeFile)
    #         writer.writerows(list)
    #     list = []
    
def createLabels():
    with open("mirai_testing_labels.csv", 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Label"])
        for i in range(514860):
            csv_writer.writerow("1")
        for i in range(19528):
            csv_writer.writerow("0")
        
    
