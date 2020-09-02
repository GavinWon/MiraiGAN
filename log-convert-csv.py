# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:47:51 2020

@author: Gavin
"""

import csv


with open('conn34.csv',"w") as outfile:
    for log_record in ParseZeekLogs("conn34.log.labeled", output_format="csv", safe_headers=False, fields=["ts","id.orig_h","id.orig_p","id.resp_h","id.resp_p"]):
        if log_record is not None:
            outfile.write(log_record + "\n")
            
with open("connlabel_malicious.csv", 'w', newline='') as file:  
    with open("connlabel_benign.csv", 'w', newline='') as file2:         
        with open("conn34.log.labeled", "r") as readfile:
            csv_writer_m = csv.writer(file)
            csv_writer_b = csv.writer(file2)
            
            for i in range(6):
                readfile.readline()
                
            l = readfile.readline().split()
            l = l[1:]
            csv_writer_m.writerow(l)
            csv_writer_b.writerow(l)
            
            readfile.readline()
            
            Lines = readfile.readlines()
            
            for line in Lines:
                if (len(l) == 1):
                    continue
                l = line.split()[1:]
                print(l)
                if (l[20] == "Benign"): #data sample is benign
                     csv_writer_b.writerow(l)
                else: #data sample is malicious
                    csv_writer_m.writerow(l)
            
           
    
print(l.shape)

with open("mirai_testing_labels.csv", 'w', newline='') as file: