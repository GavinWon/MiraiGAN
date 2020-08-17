# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:33:19 2020

@author: Gavin
"""
import csv
default_text = 'ack'
# Open the input_file in read mode and output_file in write mode
with open('All CSV\\ack.csv', 'r') as read_obj, \
        open('All CSV\\ack_new.csv', 'w', newline='') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = csv.reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = csv.writer(write_obj)
    # Read each row of the input csv file as list
    for row in csv_reader:
        # Append the default text in the row / list
        row.append(default_text)
        # Add the updated row / list to the output file
        csv_writer.writerow(row)