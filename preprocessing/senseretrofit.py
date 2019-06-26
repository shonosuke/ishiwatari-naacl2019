#!/usr/bin/env python
# encoding: utf-8
"""
Copyright (C) 2015 Sujay Kumar Jauhar <sjauhar@cs.cmu.edu>
Licenced under the Apache Licence, v2.0 - http://www.apache.org/licenses/LICENSE-2.0
"""

import sys
import getopt
import numpy
import gzip

from scipy.sparse import lil_matrix
from copy import deepcopy
from itertools import izip


help_message = '''
$ python senseretrofit.py -v <vectorsFile> -q <ontologyFile> [-o outputFile] [-n numIters] [-e epsilon] [-h]
-v or --vectors to specify path to the word vectors input file (gzip or txt files are acceptable)
-q or --ontology to specify path to the ontology (gzip or txt files are acceptable)
-o or --output to optionally set path to output word sense vectors file (<vectorsFile>.sense is used by default)
-n or --numiters to optionally set the number of retrofitting iterations (10 is the default)
-e or --epsilon to optionally set the convergence threshold (0.001 is the default)
-h or --help (this message is displayed)
'''

senseSeparator = '%'
valueSeparator = '#'


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
        

''' Read command line arguments '''
def readCommandLineInput(argv):
    try:
        try:
            #specify the possible option switches
            opts, _ = getopt.getopt(argv[1:], "hv:q:o:n:e:", ["help", "vectors=", "ontology=",
                                                              "output=", "numiters=", "epsilon="])
        except getopt.error, msg:
            raise Usage(msg)
        
        #default values
        vectorsFile = None
        ontologyFile = None
        outputFile = None
        numIters = 10
        epsilon = 0.001
        
        setOutput = False
        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage(help_message)
            elif option in ("-v", "--vectors"):
                vectorsFile = value
            elif option in ("-q", "--ontology"):
                ontologyFile = value
            elif option in ("-o", "--output"):
                outputFile = value
                setOutput = True
            elif option in ("-n", "--numiters"):
                try:
                    numIters = int(value)
                except:
                    raise Usage(help_message)
            elif option in ("-e", "--epsilon"):
                try:
                    epsilon = float(value)
                except:
                    raise Usage(help_message)
            else:
                raise Usage(help_message)
                
        if (vectorsFile==None) or (ontologyFile==None):
            raise Usage(help_message)
        else:
            if not setOutput:
                outputFile = vectorsFile + '.sense'
            return (vectorsFile, ontologyFile, outputFile, numIters, epsilon)
    
    except Usage, err:
        print str(err.msg)
        return 2


''' Read all the word vectors from file.'''
def readWordVectors(filename):
    sys.stderr.write('Reading vectors from file...\n')
    
    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'r')
    else:
        fileObject = open(filename, 'r')
    
    vectorDim = int(fileObject.readline().strip().split()[1])
    vectors = numpy.loadtxt(filename, dtype=float, comments=None, skiprows=1, usecols=range(1,vectorDim+1))
    
    wordVectors = {}
    lineNum = 0
    for line in fileObject:
        word = line.lower().strip().split()[0]
        wordVectors[word] = vectors[lineNum]
        lineNum += 1
    
    sys.stderr.write('Finished reading vectors.\n')
    
    fileObject.close()
    return wordVectors, vectorDim


''' Write word vectors to file '''
def writeWordVectors(wordVectors, vectorDim, filename):
    sys.stderr.write('Writing vectors to file...\n')
    
    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'w')
    else:
        fileObject = open(filename, 'w')
    
    fileObject.write(str(len(wordVectors.keys())) + ' ' + str(vectorDim) + '\n')
    for word in wordVectors:
        fileObject.write(word + ' ' + ' '.join(map(str, wordVectors[word])) + '\n')
    fileObject.close()
    
    sys.stderr.write('Finished writing vectors.\n')


''' Add word sense tokens to a vocabulary relevant to the input vectors.'''
def addToken2Vocab(token, vocab, vocabIndex, wordVectors):
    # check if this sense has a corresponding word vector
    if not wordVectors.has_key(token.split(senseSeparator)[0]):
        return vocabIndex
    
    # check if the sense isn't already in the vocabulary    
    if not vocab.has_key(token):
        vocab[token] = vocabIndex
        return vocabIndex + 1
    
    return vocabIndex


''' Read the subset of the ontology relevant to the input vectors.'''
def readOntology(filename, wordVectors):
    sys.stderr.write('Reading ontology from file...\n')
    
    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'r')
    else:
        fileObject = open(filename, 'r')
    
    # index all the word senses
    vocab = {}
    vocabIndex = 0    
    for line in fileObject:
        line = line.strip().split()
        for token in line:
            token = token.split(valueSeparator)[0]
            vocabIndex = addToken2Vocab(token, vocab, vocabIndex, wordVectors)
    vocabIndex += 1
    fileObject.seek(0)
    
    # create the sparse adjacency matrix of weights between senses
    adjacencyMatrix = lil_matrix((vocabIndex, vocabIndex))
    for line in fileObject:
        line = line.strip().split()
        for i in range(len(line)):
            token = line[i].split(valueSeparator)
            if vocab.has_key(token[0]):
                # find the row index
                if i == 0:
                    row = vocab[token[0]]
                # find the col index of the neighbor and set its weight
                col = vocab[token[0]]
                val = float(token[1])
                adjacencyMatrix[row, col] = val
            else:
                if i == 0:
                    break
                continue
    
    sys.stderr.write('Finished reading ontology.\n')    
    
    fileObject.close()
    # invert the vocab before returning
    vocab = {vocab[k]:k for k in vocab}
    return vocab, adjacencyMatrix.tocoo()


''' Return the maximum differential between old and new vectors
to check for convergence.'''
def maxVectorDiff(newVecs, oldVecs):
    maxDiff = 0.0
    for k in newVecs:
        diff = numpy.linalg.norm(newVecs[k] - oldVecs[k])
        if diff > maxDiff:
            maxDiff = diff
    return maxDiff


''' Run the retrofitting procedure.'''
def retrofit(wordVectors, vectorDim, senseVocab, ontologyAdjacency, numIters, epsilon):
    sys.stderr.write('Starting the retrofitting procedure...\n')
    
    # get the word types in the ontology
    ontologyWords = set([senseVocab[k].split(senseSeparator)[0] for k in senseVocab])
    # initialize sense vectors to sense agnostic counterparts
    newSenseVectors = {senseVocab[k]:wordVectors[senseVocab[k].split(senseSeparator)[0]]
                       for k in senseVocab}
    # create dummy sense vectors for words that aren't in the ontology (these won't be updated)
    newSenseVectors.update({k+senseSeparator+'0:00:00::':wordVectors[k] for k in wordVectors
                            if k not in ontologyWords})
    
    # create a copy of the sense vectors to check for convergence
    oldSenseVectors = deepcopy(newSenseVectors)
    
    # run for a maximum number of iterations
    for it in range(numIters):
        newVector = None
        normalizer = None
        prevRow = None
        sys.stderr.write('Running retrofitting iter '+str(it+1)+'... ')
        # loop through all the non-zero weights in the adjacency matrix
        for row, col, val in izip(ontologyAdjacency.row, ontologyAdjacency.col, ontologyAdjacency.data):
            # a new sense has started
            if row != prevRow:
                if prevRow:
                    newSenseVectors[senseVocab[prevRow]] = newVector/normalizer
                
                newVector = numpy.zeros(vectorDim, dtype=float)
                normalizer = 0.0
                prevRow = row
            
            # add the sense agnostic vector
            if row == col:
                newVector += val * wordVectors[senseVocab[row].split(senseSeparator)[0]]
            # add a neighboring vector    
            else:
                newVector += val * newSenseVectors[senseVocab[col]]
            normalizer += val
        
        diffScore = maxVectorDiff(newSenseVectors, oldSenseVectors)
        sys.stderr.write('Max vector differential is '+str(diffScore)+'\n')
        if diffScore <= epsilon:
            break
        oldSenseVectors = deepcopy(newSenseVectors)       
    
    sys.stderr.write('Finished running retrofitting.\n')
    
    return newSenseVectors
    

if __name__ == "__main__":
    # parse command line input
    commandParse = readCommandLineInput(sys.argv)  
    # failed command line input
    if commandParse==2:
        sys.exit(2)
    
    #try opening the specified files    
    try:
        vectors, vectorDim = readWordVectors(commandParse[0])
        senseVocab, ontologyAdjacency = readOntology(commandParse[1], vectors)
        numIters = commandParse[3]
        epsilon = commandParse[4]
    except:
        print "ERROR opening files. One of the paths or formats of the specified files was incorrect."
        sys.exit(2)
    
    # run retrofitting and write to output file   
    writeWordVectors(retrofit(vectors, vectorDim, senseVocab, ontologyAdjacency, numIters, epsilon),
                     vectorDim, commandParse[2])
    
    sys.stderr.write('All done!\n')