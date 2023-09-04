import sys

# Usage argv[1]=device1 argv[2]=device2 argv[3]=device3 ... argv[10]=device10
file=open("path_to_complete_dataset","r")

#create name 
name="data_csv/10d/data"+sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"_"+sys.argv[5]
name=name+"_"+sys.argv[6]+"_"+sys.argv[7]+"_"+sys.argv[8]+"_"+sys.argv[9]+"_"+sys.argv[10]+".csv" # saving path
file1=open(name,"w")

file1.write("timestamp,rssi,lqi,device_id\n")

i=0
for line in file:
    if(i!=0):
        line_cp=line.split(",")
        if(line_cp[4][:len(line_cp[4])-1]==sys.argv[1] or line_cp[4][:len(line_cp[4])-1]==sys.argv[2] or line_cp[4][:len(line_cp[4])-1]==sys.argv[3] or line_cp[4][:len(line_cp[4])-1]==sys.argv[4] or line_cp[4][:len(line_cp[4])-1]==sys.argv[5]):
            file1.write(line)
        elif(line_cp[4][:len(line_cp[4])-1]==sys.argv[6] or line_cp[4][:len(line_cp[4])-1]==sys.argv[7] or line_cp[4][:len(line_cp[4])-1]==sys.argv[8] or line_cp[4][:len(line_cp[4])-1]==sys.argv[9] or line_cp[4][:len(line_cp[4])-1]==sys.argv[10]):
            file1.write(line)
    i=i+1
    
