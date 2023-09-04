import sys

# Usage argv[1]=> number of elements per group       argv[2]=>first group, first device    argv[3]=>second group, first device 

file=open("path_to_complete_dataset","r")
devices_id=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154]

def look_id(val):
    for i in range(len(devices_id)):
        if(val==devices_id[i]):
            return i
    return -1


#create name 
# name="data_csv/data_group4_"+sys.argv[1]+"_"+sys.argv[2]+".csv"
name="data_csv/2d/g"+sys.argv[1]+"/data"+sys.argv[2]+"_"+sys.argv[3]+".csv"  # saving path
file1=open(name,"w")

file1.write("timestamp,rssi,lqi,device_id\n")
num=int(sys.argv[1])

id_pos1=look_id(int(sys.argv[2]))
id_pos2=look_id(int(sys.argv[3]))
# print(id_pos1,id_pos2)

id_list1=[]
id_list2=[]
for i in range(num):
    id_list1.append(devices_id[id_pos1+i])
    id_list2.append(devices_id[id_pos2+i])



i=0
for line in file:
    if(i!=0):
        line_cp=line.split(",")
        nl=line_cp[0]+line_cp[1]+","+line_cp[2]+","+line_cp[3]+","
        # print(nl)
        id=int(line_cp[4][:len(line_cp[4])-1])
        # print(id)
        if(id in id_list1):
            nl=nl+sys.argv[1]+"\n"
            # print(nl)
            file1.write(nl)
        elif(id in id_list2):
            # print(nl)
            nl=nl+sys.argv[2]+"\n"
            file1.write(nl)
    i=i+1

    
