import csv
reader=csv.reader(open(input("Enter CSV filename: "),"r"))
file=list(reader)
start=int(input("Enter 1 if there is serial number as first column or else 0: "))
heading=file[0][start:]
content=[i[start:-1] for i in file[start:] if i[-1].lower()=="yes"]
hypothesis=content[0]
for i in content[1:]:
	for j in range(len(hypothesis)):
		print(hypothesis,i)
		if hypothesis[j].lower()!=i[j].lower():
			hypothesis[j]='?'
print(hypothesis)