import sys
import os
import csv
import math
from collections import defaultdict
import random
import fnmatch
import struct


def recglob(directory, ext):
    l = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, ext):
            l.append(os.path.join(root, filename))
    return l


def load_janus_csv(filename):
    data = []
    titles = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            d = {}
            if i == 0:
                for item in row:
                    titles.append(item)
            else:
                for j, item in enumerate(row):
                    d[titles[j]] = item

                data.append(d)
    return (data, titles)


def norm(l):
    tot = 0
    for x in l:
        tot = tot + x**2
    return math.sqrt(tot)


def setup_experiment(d):
    gallery = []
    probes = []
    for item in d:
        x = random.randint(0, len(d[item]) - 1)
        gallery.append(d[item][x])
        for i in range(len(d[item])):
            if i != x:
                probes.append(d[item][i])
    return (gallery, probes)


def load_templates(directory):
    dres = {}
    files = recglob(directory, '*.tmpl')
    for filename in files:
        f = open(filename, 'rb')
        n = 128
        l = list(struct.unpack('f' * n, f.read(4 * n)))
        lres = []
        for x in l:
            lres.append(x)
        key = filename.split('/')[-1].replace('.tmpl', '')
        dres[key] = lres

    return dres


def find_rank(probe, ranking, d):
    for i, item in enumerate(ranking):
        if d[probe] == d[item]:
            return i
    return len(ranking)


def main():
    (data, _) = load_janus_csv(sys.argv[1])
    d = defaultdict(list)
    identity = {}
    for item in data:
        if item['TEMPLATE_ID'] not in d[item['SUBJECT_ID']]:
            d[item['SUBJECT_ID']].append(item['TEMPLATE_ID'])
            identity[item['TEMPLATE_ID']] = item['SUBJECT_ID']

    queries = 0
    stat = defaultdict(int)

    d_templates = load_templates(sys.argv[2])

    for _ in range(200):
        (gallery, probes) = setup_experiment(d)
        galfeats = [d_templates[x] for x in gallery]
        probefeats = [d_templates[x] for x in probes]
        results = []
        for p in probefeats:
            res = []
            for i, g in enumerate(galfeats):
                np = norm(p)
                ng = norm(g)
                sim = sum([a * b for (a, b) in zip(p, g)]) / (np * ng)
                res.append((sim, i))
            res = sorted(res, reverse=True)
            results.append(res)

        results_template = []
        for item in results:
            r = []
            for (_, index) in item:
                r.append(gallery[index])
            results_template.append(r)

        for (item, probe) in zip(results_template, probes):
            rank = find_rank(probe, item, identity)
            for i in range(rank, len(gallery)):
                stat[i] = stat[i] + 1
            queries = queries + 1

    for i in sorted(stat.keys()):
        print 'rank: %d, accuracy: %f'%(i+1, stat[i]/float(queries))


if __name__ == "__main__":
    main()
