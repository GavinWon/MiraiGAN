# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:47:51 2020

@author: Gavin
"""

import csv
import pandas
import fileinput

def covertLogToCSV(): #72317 total samples
    with open("conn.csv", 'w', newline='') as file:  
        # with open("connlabel_benign.csv", 'w', newline='') as file2:         
        with open("conn.log", "r") as readfile:
            csv_writer_m = csv.writer(file)
            # csv_writer_b = csv.writer(file2)
            
            for i in range(6):
                readfile.readline()
                
            l = readfile.readline().split()
            l = l[1:]
            l.append("Label")
            csv_writer_m.writerow(l)
            
            # csv_writer_b.writerow(l)
            
            readfile.readline()
            
            Lines = readfile.readlines()
            count = 0
            for line in Lines:
                l = line.split()[:]
                l.append("Mirai")
                if "-" in l:
                    count += 1
                # if (len(l) < 21):
                #     continue
                csv_writer_m.writerow(l)
                # if (l[21] == "Benign"): #data sample is benign
                #      csv_writer_b.writerow(l)
                # else: #data sample is malicious
                #     csv_writer_m.writerow(l)
            print(count)
            

def removeCol():
    with open("connlabel_malicious.csv","r") as source:
        rdr = csv.reader( source )
        with open("connlabel_malicious_new.csv","w", newline="") as result:
            count = 0
            wtr = csv.writer( result )
            for r in rdr:
                if (count % 1000000 == 0):
                    print(count)
                wtr.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[11], r[14], r[15], r[16], r[17], r[18],  r[19], r[21], r[22] ] )  
                count += 1
                

     

def removeRows():
    #remove rows with - in the row -->66893 left (5425 removed)
    with open('conn.csv', 'r') as read_obj:
        with open('conn1.csv', 'w', newline='') as write_obj:
            csv_reader = csv.reader(read_obj)
            csv_writer = csv.writer(write_obj)
            count = 0
            seen = set()
            for row in csv_reader:
                check = row[:-1] #dont consider the last column
                if ("-" not in check):
                    csv_writer.writerow(row)
                    count += 1
            print(count)
            
    #remove duplicates --> no duplicates removed
    count = 0
    with open('conn1.csv','r') as in_file:
        with open('conn2.csv','w', newline="") as out_file:
            csv_writer = csv.writer(out_file)
            seen = set() # set for fast O(1) amortized lookup
            for line in in_file:
                line = tuple(line.split(",")) 
                check = line[2:]
                if check in seen: continue # skip duplicate
                seen.add(check)
                csv_writer.writerow(line)
        print(count)    
    
    
def createDataSet():
    with open('connlabel_malicious_new2.csv', 'r') as read_obj:
        with open('malicious.csv', 'w', newline='') as write_obj:
            csv_reader = csv.reader(read_obj)
            csv_writer = csv.writer(write_obj)
            
            for i in range(10000000):
                csv_writer.writerow(next(csv_reader))
                
def addLabels():
    
        
import pandas as pd
df = pd.read_csv('connlabel_benign_new2.csv')    
df.head()

db = pd.read_csv('34\\connlabel_benign34_new2.csv')
            
        