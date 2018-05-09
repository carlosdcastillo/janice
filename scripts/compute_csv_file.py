import os
import struct
import sys
import fnmatch

def recglob(directory,ext):
    l = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, ext):
            l.append(os.path.join(root, filename))
    return l

def main():
    directory = sys.argv[1]
    outputcsv = sys.argv[2]

    files = sorted(recglob(directory, '*.tmpl'))
    fout = open(outputcsv, 'w')
    header = ['FILE']
    for i in range(1,129):
        header.append('DEEPFEATURE_%d'%i)
    fout.write(','.join(header)+'\n')
    lw = []
    for filename in files:
        f = open(filename,'rb')
        n = 128
        l = list(struct.unpack('f'*n, f.read(4*n)))
        lres = []
        lres.append(filename)
        for x in l:
            lres.append(str(x))
        lw.append(','.join(lres))
    fout.write('\n'.join(lw))

if __name__ == "__main__":
    main()
