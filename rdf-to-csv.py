#!/usr/bin/python
import sys
import argparse
from rdflib.graph import Graph
import csv


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
    writer.writerow(row)

