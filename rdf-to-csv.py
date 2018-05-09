#!/usr/bin/python
from __future__ import print_function
import sys
import argparse
from rdflib.graph import Graph
import csv

def to_utf8(lst):
    return [unicode(elem).encode('utf-8') for elem in lst]

parser = argparse.ArgumentParser(description='Avoiding False Negative Samples on Link Prediction')


# Required positional argument
parser.add_argument('file',
                    help='Filename')

parser.add_argument('--format', default="nt",
                    help='RDF Format of the input (passed to rdflib parser)')


args = parser.parse_args()


graph = Graph()
graph.parse(args.file, format=args.format)

writer = csv.writer(sys.stdout, dialect='excel-tab')

for s, p, o in graph:
    row = [ "" + s, "" + p, "" + o ]
    try:
        writer.writerow(to_utf8(row))
    except Exception, e:
         print('Error with row' + str(row) + ':' + str(e), file=sys.stderr)
