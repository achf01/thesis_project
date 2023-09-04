import sys

# Usage argv[1]=device1 argv[2]=device2 
file=open("path_to_complete_dataset","r")

#create name 
name="data_csv/data"+sys.argv[1]+"_"+sys.argv[2]+".csv" # saving path
file1=open(name,"w")

file1.write("timestamp,rssi,lqi,device_id\n")

i=0
for line in file:
    if(i!=0):
        line_cp=line.split(",")
        if(line_cp[4][:len(line_cp[4])-1]==sys.argv[1] or line_cp[4][:len(line_cp[4])-1]==sys.argv[2]):
            file1.write(line)
    i=i+1
    
