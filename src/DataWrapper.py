import gzip
import csv

class DataWrapper(object):
    """ Read in the processed data that is already tokenized by space"""
    def __init__(self, filename):
        print("Read in %s" % filename)
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, 'r'):
            yield line.strip().split()


class DataWrapperCSV(object):
    """ Read in the csv file"""
    def __init__(self, filename):
        print("Read in %s" % filename)
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'r') as f1:
            reader = csv.DictReader(f1)
            for row in reader:
                yield row


class DataWrapperGzipCSV(object):
    """ Read in gzip csv file"""
    def __init__(self, filename):
        print("Read in %s" % filename)
        self.filename = filename

    def __iter__(self):
        with gzip.open(self.filename, 'rt') as f1:
            reader = csv.DictReader(f1)
            for row in reader:
                yield row



class DataWrapperGzip (object):
    """ Read in the processed data that is already tokenized by space. Gzip version."""
    def __init__(self, filenames):
        self.filenames = filenames
        print(filenames)

    def __iter__(self):
        for filename in self.filenames:
            with gzip.open(filename,'rt') as fin:      
                for line in fin:   
                    yield line.strip().split()  

class DataWrapperGzipMulti (object):
    """ Read in the processed data that is already tokenized by space and associated metadata. Gzip version."""
    def __init__(self, filenames1, filenames2):
        """
        Two lists of filenames (content + metadata) """
        self.filenames1 = filenames1
        self.filenames2 = filenames2
        
        print(filenames1)
        print(filenames2)
        

    def __iter__(self):
        for idx1, filename1 in enumerate(self.filenames1):
            with gzip.open(filename1,'rt') as fin1:
                 with gzip.open(self.filenames2[idx1],'rt') as fin2:
                    line = fin1.readline()
                    while line:
                        yield(line.strip().split(), fin2.readline())
                        line = fin1.readline()
                        