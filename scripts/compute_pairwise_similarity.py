import os
import math
import csv
import sys

def load_janus_csv(filename):
    data = [] 
    titles = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for i,row in enumerate(reader):
            d = {}
            if i==0:
                for item in row:
                    titles.append(item)
            else:
                for j,item in enumerate(row):
                    d[titles[j]] = item

                data.append(d)
        #print titles
    return (data, titles)

def norm(l):
    tot = 0
    for x in l:
        tot = tot + x**2
    return math.sqrt(tot)


def main():
    (data, titles) = load_janus_csv(sys.argv[1])
    fout = open('similarity_matrix.csv','w')
    tit = []
    tit.append('FILENAME')
    for item in data:
        tit.append(item['FILE'])
    fout.write(','.join(tit)+'\n')

    for item in data:
        l = []
        l.append(item['FILE'])

        for item2 in data:
            d1 = []
            for i in range(1,129):
                d1.append(float(item['DEEPFEATURE_%d'%i]))
            d2 = []
            for i in range(1,129):
                d2.append(float(item2['DEEPFEATURE_%d'%i]))

            nd1 = norm(d1)
            nd2 = norm(d2)
            tot = 0
            for (a,b) in zip(d1,d2):
                tot = tot + a*b
            tot  = tot / (nd1*nd2)
            l.append(str(tot))
        fout.write(','.join(l)+'\n')

if __name__ == "__main__":
    main()
