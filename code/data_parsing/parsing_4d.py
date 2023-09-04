import sys

# Usage argv[1]=device1 argv[2]=device2 argv[3]=device3 argv[4]=device4 
file=open("path_to_complete_dataset","r") 

#create name 
name="data_csv/4d/data"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+".csv" # saving path
file1=open(name,"w")

file1.write("timestamp,rssi,lqi,device_id\n")

i=0
for line in file:
    if(i!=0):
        line_cp=line.split(",")
        if(line_cp[4][:len(line_cp[4])-1]==sys.argv[1] or line_cp[4][:len(line_cp[4])-1]==sys.argv[2] or line_cp[4][:len(line_cp[4])-1]==sys.argv[3] or line_cp[4][:len(line_cp[4])-1]==sys.argv[4]):
            file1.write(line)
    i=i+1
    
