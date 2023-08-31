import sys

if len(sys.argv) < 2:
    print(f"Usage {sys.argv[0]} inputFile")
    exit(1)

data=""
file=open(sys.argv[1], 'r')
file1=open("data.csv", "a")
for line in file:
    if("rssi" in line and "lqi" in line):
        line=line.split()
        data="'"+line[0][1:]+" "+line[1][:len(line[1])-1]+ "',"+ (line[10])+","+(line[12][:len(line[12])-1])+","+line[3][:len(line[3])-8]+"\n"
        file1.write(data)

# print(data)
