from sklearn import preprocessing
# 数据，二维
dataset = []
label = ["normal","ipsweep","mscan","nmap","portsweep","saint","satan","apache2","back","land","mailbomb",
         "neptune","pod","processtable","smurf","teardrop","udpstorm","buffer_overflow","httptunnel",
         "loadmodule","perl","ps","rootkit","sqlattack","xterm","ftp_write","guess_passwd","imap",
         "multihop","named","phf","sendmail","snmpgetattack","snmpguess","spy","warezclient","warezmaster",
         "worm","xlock","xsnoop"]
le = preprocessing.LabelEncoder()
le.fit(label)
try:
    f = open('D:\\Users\\liuxudong\\PycharmProjects\\L-layers\\dataset\\ceshi\\corrected(122)(0-1).txt', 'r')
    newf = open(r'dataset/corrected(lable 1-40).txt', 'w')
    #print(f.read())
    a = f.read()
    data = a.splitlines()
    print("len", len(data))
    for i in data:
        #print(i.split(','))
        b = i.split(',')
        if b[-1] in label:
            b[-1] = str(label.index(b[-1]) + 1)
        #b = list(map(float, b))
        dataset.append(b)
        newf.write(str(b).replace("'", "").replace("[", "").replace("]", "").replace(" ", "").replace(","," ")+"\n")
    print(dataset[0])
finally:
    if f:
        f.close()
        newf.close()


