#!/usr/bin/env python3
#-*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: If you have trained the module_train on a training set (terms associated with concept(s)),
    you can do here a prediction of normalization with a test set (new terms without pre-association with concept).
    If you want to cite this work in your publication or to have more details:
    Ongoing...

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


#######################################################################################################
# Import modules & set up logging
#######################################################################################################

from io import open
from sys import stderr, path
from optparse import OptionParser
import json
import gzip
from os.path import dirname, exists, abspath

path.append(dirname(abspath(__file__))+'/../utils')
import word2term, onto
import numpy
import gensim
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from tensorflow.keras import layers, models


#######################################################################################################
# Utils
#######################################################################################################

def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        f = open(filename, encoding='utf-8')
    result = json.load(f)
    f.close()
    return result;



def metric_internal(metric):
    if metric == 'cosine':
        return 'euclidean'
    if metric == 'cosine-brute':
        return 'cosine'
    return metric


def metric_norm(metric, concept_vectors):
    if metric == 'cosine':
        return normalize(concept_vectors)
    return concept_vectors


def metric_sim(metric, d, vecTerm, vecConcept):
    if metric == 'cosine':
        return 1 - cosine(vecTerm, vecConcept)
    if metric == 'cosine-brute':
        return 1 - d
    return 1 / d



class VSONN(NearestNeighbors):
    def __init__(self, vso, metric):
        NearestNeighbors.__init__(self, algorithm='auto', metric=metric_internal(metric))
        self.original_metric = metric
        self.vso = vso
        self.concepts = tuple(vso.keys())
        self.concept_vectors = list(vso.values())
        self.fit(metric_norm(metric, self.concept_vectors))

    def nearest_concept(self, vecTerm):
        r = self.kneighbors([vecTerm], 1, return_distance=True)
        #stderr.write('r = %s\n' % str(r))
        d = r[0][0][0]
        idx = r[1][0][0]
        return self.concepts[idx], metric_sim(self.original_metric, d, vecTerm, self.concept_vectors[idx])



#######################################################################################################
# Main methods
#######################################################################################################

######
# Single Layer Feedforward Neural Network (SLFNN) predictor:
######
def SLFNNpredictor(vst_onlyTokens, dl_terms, vso, transformationParam, metric, symbol='___'):
    """
    Description: From a calculated linear projection from the training module, applied it to predict a concept for each
        terms in parameters (dl_terms).
    :param vst_onlyTokens: An initial VST containing only tokens and associated vectors.
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param vso: A VSO (dict() -> {"id" : [vector], ...}
    :param transformationParam: LinearRegression object from Sklearn. Use the one calculated by the training module.
    :param symbol: Symbol delimiting the different token in a multi-words term.
    :return: A list of tuples containing : ("term form", "term id", "predicted concept id") and a list of unknown tokens
        containing in the terms from dl_terms.
    """
    lt_predictions = list()

    vstTerm, l_unknownToken = word2term.wordVST2TermVST(vst_onlyTokens, dl_terms)

    result = dict()

    vsoTerms = dict()
    vsoNN = VSONN(vso, metric)
    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        x = vstTerm[termForm].reshape(1, -1)

        vsoTerms[termForm] = transformationParam.predict(x)[0]

        result[termForm] = vsoNN.nearest_concept(vsoTerms[termForm])

    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        cat, sim = result[termForm]
        prediction = (termForm, id_term, cat, sim)
        lt_predictions.append(prediction)

    return lt_predictions, l_unknownToken



######
# Shallow-CNN predictor:
######
def SCNNpredictor(vst_onlyTokens, dl_terms, vso, transformationParam, metric, phraseMaxSize, symbol='___'):
    """
    Description: From a calculated linear projection from the training module, applied it to predict a concept for each
        terms in parameters (dl_terms).
    :param vst_onlyTokens: An initial VST containing only tokens and associated vectors.
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param vso: A VSO (dict() -> {"id" : [vector], ...}
    :param transformationParam: LinearRegression object from Sklearn. Use the one calculated by the training module.
    :param symbol: Symbol delimiting the different token in a multi-words term.
    :return: A list of tuples containing : ("term form", "term id", "predicted concept id") and a list of unknown tokens
        containing in the terms from dl_terms.
    """
    lt_predictions = list()
    result = dict()
    vsoTerms = dict()

    sizeVST = word2term.getSizeOfVST(vst_onlyTokens)

    vsoNN = VSONN(vso, metric)
    for id_term in dl_terms.keys():
        x = numpy.zeros((1, phraseMaxSize, sizeVST))
        for i, token in enumerate(dl_terms[id_term]):
            try:
                x[i] = vst_onlyTokens[token]
            except:
                pass

        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        vsoTerms[termForm] = transformationParam.predict(x)[0][0]

        result[termForm] = vsoNN.nearest_concept(vsoTerms[termForm])

    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        cat, sim = result[termForm]
        prediction = (termForm, id_term, cat, sim)
        lt_predictions.append(prediction)

    return lt_predictions



######
# C-Norm predictor
######
def CNorm_Predictor(vst_onlyTokens, dl_terms, vso, transformationParam, metric, phraseMaxSize, symbol='___'):

    lt_predictions = list()
    result = dict()
    vsoTerms = dict()

    sizeVST = word2term.getSizeOfVST(vst_onlyTokens)

    vsoNN = VSONN(vso, metric)
    for id_term in dl_terms.keys():
        x_CNN = numpy.zeros((1, phraseMaxSize, sizeVST))
        x_MLP = numpy.zeros((1, sizeVST))
        for i, token in enumerate(dl_terms[id_term]):
            try:
                x_CNN[0][i] = vst_onlyTokens[token]
                x_MLP[0] += vst_onlyTokens[token]
            except:
                pass
        if len(dl_terms[id_term]) == 0:
            pass
        else:
            x_MLP[0] = x_MLP[0] / len(dl_terms[id_term])

        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        vsoTerms[termForm] = transformationParam.predict([x_MLP, x_CNN])[0][0]

        result[termForm] = vsoNN.nearest_concept(vsoTerms[termForm])

    for id_term in dl_terms.keys():
        termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
        cat, sim = result[termForm]
        prediction = (termForm, id_term, cat, sim)
        lt_predictions.append(prediction)

    return lt_predictions



######
# Run class:
######

class Predictor(OptionParser):
    def __init__(self):
        OptionParser.__init__(self, usage='usage: %prog [options]')
        self.add_option('--word-vectors', action='store', type='string', dest='word_vectors', help='path to word vectors file as produced by word2vec')
        self.add_option('--word-vectors-bin', action='store', type='string', dest='word_vectors_bin', help='path to word vectors binary file as produced by word2vec')
        self.add_option('--ontology', action='store', type='string', dest='ontology', help='path to ontology file in OBO format')
        self.add_option('--terms', action='append', type='string', dest='terms', help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--factor', action='append', type='float', dest='factors', default=[], help='parent concept weight factor (default: 1.0)')
        self.add_option('--model', action='append', type='string', dest='model', help='path to the model from a training')
        self.add_option('--output', action='append', type='string', dest='output', help='file where to write predictions')

        self.add_option('--metric', action='store', type='string', dest='metric', default='cosine', help='distance metric to use (default: %default)')

        self.add_option('--sieve-slfnn', action='store', type='string', dest='slfnn_model', help='path to the model for Shallow CNN from another training (Sieve S-CNN>SLFNN model).')
        self.add_option('--threshold', action='store', type='float', dest='threshold', help='threshold value for SIEVE method, between 0.0 (all S-CNN) and 1.0 (all SLFNN).')


    def run(self):
        options, args = self.parse_args()
        if len(args) > 0:
            raise Exception('stray arguments: ' + ' '.join(args))
        if options.word_vectors is None and options.word_vectors_bin is None:
            raise Exception('missing either --word-vectors or --word-vectors-bin')
        if options.word_vectors is not None and options.word_vectors_bin is not None:
            raise Exception('incompatible --word-vectors or --word-vectors-bin')        
        if options.ontology is None:
            raise Exception('missing --ontology')
        if not(options.terms):
            raise Exception('missing --terms')
        if not(options.model):
            raise Exception('missing --model')
        if not(options.output):
            raise Exception('missing --output')
        if len(options.terms) != len(options.model):
            raise Exception('there must be the same number of --terms and --model')
        if len(options.terms) != len(options.output):
            raise Exception('there must be the same number of --terms and --output')
        if len(options.factors) > len(options.terms):
            raise Exception('there must be at least as many --terms as --factor')


        if len(options.factors) < len(options.terms):
            n = len(options.terms) - len(options.factors)
            stderr.write('defaulting %d factors to 1.0\n' % n)
            stderr.flush()
            options.factors.extend([1.0]*n)

        if options.word_vectors is not None:
            stderr.write('loading word embeddings: %s\n' % options.word_vectors)
            stderr.flush()
            word_vectors = loadJSON(options.word_vectors)
        elif options.word_vectors_bin is not None:
            stderr.write('loading word embeddings: %s\n' % options.word_vectors_bin)
            stderr.flush()
            model = gensim.models.Word2Vec.load(options.word_vectors_bin)
            word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in model.wv[k])) for k in model.wv.vocab.keys())

        stderr.write('loading ontology: %s\n' % options.ontology)
        stderr.flush()
        ontology = onto.loadOnto(options.ontology)


        for terms_i, model_i, output_i, factor_i in zip(options.terms, options.model, options.output, options.factors):

            vso = onto.ontoToVec(ontology, factor_i)

            stderr.write('loading terms: %s\n' % terms_i)
            stderr.flush()
            terms = loadJSON(terms_i)

            stderr.write('predicting...\n')
            stderr.flush()

            #metric = "cosine"


            if options.slfnn_model is not None:

                stderr.write('\nYou\'re using SIEVE prediction method because --sieve-mlp parameter is not None (%s).\n' % options.slfnn_model)
                stderr.flush()

                print("Threshold: ", options.threshold)
                if options.threshold is None:
                    raise Exception('A threshold parameter is needed (--threshold) to run Sieve method (between 0.0 and 1.0).')

                stderr.write('\nloading Tensorflow model (SLFNN): %s\n' % options.slfnn_model)
                stderr.flush()
                model_SLFNN = models.load_model(options.slfnn_model)

                stderr.write('SLFNN prediction...\n')
                stderr.flush()
                predictionLP, _ = SLFNNpredictor(word_vectors, terms, vso, model_SLFNN, options.metric, symbol='___')

                stderr.write('\nloading Tensorflow model (Shallow CNN): %s\n' % model_i)
                stderr.flush()
                model_SCNN = models.load_model(model_i)

                stderr.write('Shallow CNN prediction...\n')
                stderr.flush()
                predictionCNN = SCNNpredictor(word_vectors, terms, vso, model_SCNN, options.metric, 15)

                stderr.write('\nSieving...')
                stderr.flush()
                prediction = list()
                for termForm, term_id, concept_id, similarity in predictionCNN:
                    if similarity <= options.threshold:
                        for termForm_MLP, term_id_MLP, concept_id_MLP, similarity_MLP in predictionLP:
                            if term_id_MLP == term_id:
                                current_prediction = (termForm_MLP, term_id_MLP, concept_id_MLP, similarity)
                                prediction.append(current_prediction)
                    else:
                        current_prediction = (termForm, term_id, concept_id, similarity)
                        prediction.append(current_prediction)


            else:

                stderr.write('loading Tensorflow model: %s\n' % model_i)
                stderr.flush()
                trained_model = models.load_model(model_i)

                try:
                    stderr.write('\nTrying to run Shallow CNN predictor...\n')
                    stderr.flush()

                    prediction = SCNNpredictor(word_vectors, terms, vso, trained_model, options.metric, 15)


                except:
                    try:
                        stderr.write('\nS-CNN aborted. Trying to run SLFNN predictor...\n')
                        stderr.flush()

                        prediction, _ = SLFNNpredictor(word_vectors, terms, vso, trained_model, options.metric, symbol='___')


                    except:
                        stderr.write('\nSLFNN aborted. Trying to run C-Norm predictor...\n')
                        stderr.flush()

                        prediction = CNorm_Predictor(word_vectors, terms, vso, trained_model, options.metric, 15)



            stderr.write('\nwriting predictions: %s\n' % output_i)
            stderr.flush()
            f = open(output_i, 'w')
            dl_prediction = dict()
            for _, term_id, concept_id, similarity in prediction:
                f.write('%s\t%s\t%f\n' % (term_id, concept_id, similarity))
                dl_prediction[term_id] = [concept_id]
            f.close()



#######################################################################################################
# Test section
#######################################################################################################

if __name__ == '__main__':

    try:
        Predictor().run()

    except:

        # Path to test data:
        mentionsFilePath = "../test/DATA/trainingData/terms_dev.json"
        SSO_path = "../test/DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"

        # load input data:
        print("\nLoading data...")
        extractedMentionsFile = open(mentionsFilePath, 'r')
        dl_testedTerms = json.load(extractedMentionsFile)
        extractedMentionsFile.close()

        # Building of concept embeddings and training:
        print("Loading Semantic Space of the Ontology (SSO)...")
        SSO_file = open(SSO_path, "r")
        SSO = json.load(SSO_file)
        print("SSO loaded.\n")

        # Loading embeddings:
        print("Loading token embeddings...")
        vst_Path = "../test/DATA/wordEmbeddings/VST_count0_size100_iter50.model"
        vst_onlyTokensModel = gensim.models.Word2Vec.load(vst_Path)
        vst_onlyTokens = dict((k, list(numpy.float_(npf32) for npf32 in vst_onlyTokensModel.wv[k])) for k in vst_onlyTokensModel.wv.vocab.keys())
        print("Embeddings loaded.")


        print("\n\nAll methods test prediction...")
        metric = "cosine"


        stderr.write('\nSLFNN predicting...\n')
        stderr.flush()
        regmatPath = "../test/DATA/learnedHyperparameters/SLFNN"
        SLFNNmodel = models.load_model(regmatPath)
        print("     Data loaded.")
        SLFNNpredictions, _ = SLFNNpredictor(vst_onlyTokens, dl_testedTerms, SSO, SLFNNmodel, metric, symbol = '___')
        print("prediction done.\n")


        stderr.write('\nShallow-CNN predicting...\n')
        stderr.flush()
        regmatPath = "../test/DATA/learnedHyperparameters/SCNN"
        SCNNmodel = models.load_model(regmatPath)
        print("     Data loaded.")
        SCNNpredictions = SCNNpredictor(vst_onlyTokens, dl_testedTerms, SSO, SCNNmodel, metric, 15, symbol = '___')
        print("prediction done.\n")


        stderr.write('\nC-Norm predicting...\n')
        stderr.flush()
        regmatPath = "../test/DATA/learnedHyperparameters/CNorm"
        CNorm_model = models.load_model(regmatPath)
        print("     Data loaded.")
        lt_predictions = CNorm_Predictor(vst_onlyTokens, dl_testedTerms, SSO, CNorm_model, metric, 15, symbol='___'  )
        print("prediction done.\n")


        stderr.write('Sieve predicting...\n')
        stderr.flush()
        threshold = 0.5
        stderr.flush()
        prediction = list()
        for termForm, term_id, concept_id, similarity in SCNNpredictions:
            if similarity <= threshold:
                for termForm_MLP, term_id_MLP, concept_id_MLP, similarity_MLP in SLFNNpredictions:
                    if term_id_MLP == term_id:
                        current_prediction = (termForm_MLP, term_id_MLP, concept_id_MLP, similarity)
                        prediction.append(current_prediction)
            else:
                current_prediction = (termForm, term_id, concept_id, similarity)
                prediction.append(current_prediction)


        print("Prediction test done.\n")


