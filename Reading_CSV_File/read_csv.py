import csv
with open(input("CSV File Name: "), 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		print(row)