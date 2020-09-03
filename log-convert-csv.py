# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:47:51 2020

@author: Gavin
"""

import csv

def covertLogToCSV():
    with open("connlabel34.csv", 'w', newline='') as file:  
        # with open("connlabel_benign.csv", 'w', newline='') as file2:         
        with open("conn34.log.labeled", "r") as readfile:
            csv_writer_m = csv.writer(file)
            # csv_writer_b = csv.writer(file2)
            
            for i in range(6):
                readfile.readline()
                
            l = readfile.readline().split()
            l = l[1:]
            csv_writer_m.writerow(l)
            # csv_writer_b.writerow(l)
            
            readfile.readline()
            
            Lines = readfile.readlines()
            count = 0
            for line in Lines:
                l = line.split()
                if (count % 1000000 == 0):
                    print(count)
                if (len(l) < 21):
                    continue
                csv_writer_m.writerow(l)
                # if (l[21] == "Benign"): #data sample is benign
                #      csv_writer_b.writerow(l)
                # else: #data sample is malicious
                #     csv_writer_m.writerow(l)
                count += 1
            

def removeCol():
    with open("connlabel_benign.csv","r") as source:
        rdr = csv.reader( source )
        with open("connlabel_benign_new.csv","w") as result:
            wtr = csv.writer( result )
            for r in rdr:
                wtr.writerow( (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[11], r[14], r[15], r[16], r[17], r[18],  r[19], r[21], r[22] )   )  
     
    
print(l.shape)

with open("mirai_testing_labels.csv", 'w', newline='') as file: