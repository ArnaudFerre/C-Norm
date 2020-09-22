#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coding: utf-8


"""
Author: Arnaud Ferré
Mail: arnaud.ferre.pro@gmail.com
Description: Training module for C-Norm method

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

from sys import stderr, path, exit
from optparse import OptionParser
import json
import gzip
from os.path import dirname, exists, abspath
from os import makedirs

import numpy
import gensim
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, backend

path.append(dirname(abspath(__file__))+"/../utils/")
import word2term, onto

#######################################################################################################
# Utils
#######################################################################################################

# Normalization of token embeddings:
def normalizeEmbedding(vst_onlyTokens):

    for token in vst_onlyTokens.keys():
        vst_onlyTokens[token] = vst_onlyTokens[token] / numpy.linalg.norm(vst_onlyTokens[token])

    return vst_onlyTokens


#
def getMaxSentenceSize(dl_trainingTerms):
    maxLength = 0
    for mentionID in dl_trainingTerms.keys():
        currentSize = len(dl_trainingTerms[mentionID])
        if currentSize > maxLength:
            maxLength = currentSize
    return  maxLength


#
def getMatrix(dl_terms, vstTerm, dl_associations, vso, symbol="___"):
    """
    Description: Create the 2 training matrix in respect to their association. Each row of the matrix correspond to a
        vector (associated with a term or a concept). For the same number of a row, there is the vector of a term and
        for the other a vector of the associated concept ; (n_samples, n_features) and (n_sample, n_targets).
    :param dl_terms: A dictionnary with id of terms for key and raw form of terms in value.
    :param vstTerm: VST with only the terms and the vectors contained in the dl_terms.
    :param dl_associations: The training set associating the id of terms and the if of concepts (NB: the respect of these
        IDs is the responsability of the user).
    :param vso: A VSO, that is a dictionary with id of concept (<XXX_xxxxxxxx: Label>) as keys and a numpy vector in value.
    :param symbol: Symbol delimiting the different token in a multi-words term.
    :return: Two matrix, one for the vectors of terms and another for the associated vectors of concepts.
    """

    nbTerms = len(dl_terms.keys())
    sizeVST = word2term.getSizeOfVST(vstTerm)
    sizeVSO = word2term.getSizeOfVST(vso)
    X_train = numpy.zeros((nbTerms, sizeVST))
    Y_train = numpy.zeros((nbTerms, sizeVSO))

    i = 0
    for id_term in dl_associations.keys():
        # stderr.write('id_term = %s\n' % str(id_term))
        # stderr.write('len(dl_associations[id_term]) = %d\n' % len(dl_associations[id_term]))
        for id_concept in dl_associations[id_term]:
            # stderr.write('id_concept = %s\n' % str(id_concept))
            termForm = word2term.getFormOfTerm(dl_terms[id_term], symbol)
            # stderr.write('termForm = %s\n' % str(termForm))
            X_train[i] = vstTerm[termForm]
            Y_train[i] = vso[id_concept]
            i += 1
            break
    return X_train, Y_train


# CHoose with getMatricForCNN...?
def prepare2D_data(vst_onlyTokens, dl_terms, dl_associations, vso, phraseMaxSize):

    #ToDo: Keep a constant size of the input matrix between train and prediction
    nbTerms = len(dl_terms.keys())
    sizeVST = word2term.getSizeOfVST(vst_onlyTokens)
    sizeVSO = word2term.getSizeOfVST(vso)

    X_train = numpy.zeros((nbTerms, phraseMaxSize, sizeVST))
    Y_train = numpy.zeros((nbTerms, 1, sizeVSO))

    l_unkownTokens = list()
    l_uncompleteExpressions = list()

    for i, id_term in enumerate(dl_associations.keys()):
        # stderr.write('id_term = %s\n' % str(id_term))
        # stderr.write('len(dl_associations[id_term]) = %d\n' % len(dl_associations[id_term]))

        for id_concept in dl_associations[id_term]:
            Y_train[i][0] = vso[id_concept]
            for j, token in enumerate(dl_terms[id_term]):
                if j < phraseMaxSize:
                    if token in vst_onlyTokens.keys():
                        X_train[i][j] = vst_onlyTokens[token]
                    else:
                        l_unkownTokens.append(token)
                else:
                    l_uncompleteExpressions.append(id_term)
            break # Because it' easier to keep only one concept per mention (mainly to calculate size of matrix).
            # ToDo: switch to object to include directly size with these structures.

    return X_train, Y_train, l_unkownTokens, l_uncompleteExpressions


#
def loadJSON(filename):
    if filename.endswith('.gz'):
        f = gzip.open(filename)
    else:
        # f = open(filename, encoding='utf-8')
        f = open(filename, "r", encoding="utf-8")
    result = json.load(f)
    f.close()
    return result;




#######################################################################################################
# Main methods
#######################################################################################################

######
# Single Layer Feedforward Neural Network (SLFNN):
######
def SLFNN(vst_onlyTokens, dl_terms, dl_associations, vso, nbEpochs=100, batchSize=64):

    vstTerm, l_unknownToken = word2term.wordVST2TermVST(vst_onlyTokens, dl_terms)
    data, labels = getMatrix(dl_terms, vstTerm, dl_associations, vso, symbol="___")

    inputSize = data.shape[1]
    ontoSpaceSize = labels.shape[1]

    model = models.Sequential()
    model.add(layers.Dense(units=ontoSpaceSize, use_bias=True, kernel_initializer=initializers.GlorotUniform(), input_shape=(inputSize,)))
    model.summary()

    model.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=[metrics.CosineSimilarity(), metrics.MeanSquaredError()])
    model.fit(data, labels, epochs=nbEpochs, batch_size=batchSize)

    return model, vso, l_unknownToken




######
# Shallow CNN (S-CNN):
######
def SCNN(vst_onlyTokens, dl_terms, dl_associations, vso,
         nbEpochs=150, batchSize=64,
         l_numberOfFilters=[4000], l_filterSizes=[1],
         phraseMaxSize=15):

    data, labels, l_unkownTokens, l_uncompleteExpressions = prepare2D_data(vst_onlyTokens, dl_terms, dl_associations, vso, phraseMaxSize)

    embeddingSize = data.shape[2]
    ontoSpaceSize = labels.shape[2]

    inputLayer = Input(shape=(phraseMaxSize, embeddingSize))

    l_subLayers = list()
    for i, filterSize in enumerate(l_filterSizes):

        convLayer = (layers.Conv1D(l_numberOfFilters[i], filterSize, strides=1, kernel_initializer=initializers.GlorotUniform()))(inputLayer)

        outputSize = phraseMaxSize - filterSize + 1
        pool = (layers.MaxPool1D(pool_size=outputSize))(convLayer)

        activationLayer = (layers.LeakyReLU(alpha=0.3))(pool)

        l_subLayers.append(activationLayer)

    if len(l_filterSizes) > 1:
        concatenateLayer = (layers.Concatenate(axis=-1))(l_subLayers)  # axis=-1 // concatenating on the last dimension
    else:
        concatenateLayer = l_subLayers[0]

    convModel = Model(inputs=inputLayer, outputs=concatenateLayer)
    fullmodel = models.Sequential()
    fullmodel.add(convModel)

    fullmodel.add(layers.Dense(ontoSpaceSize, kernel_initializer=initializers.GlorotUniform()))

    fullmodel.summary()
    fullmodel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=[metrics.CosineSimilarity(), metrics.MeanSquaredError()])
    fullmodel.fit(data, labels, epochs=nbEpochs, batch_size=batchSize)

    return fullmodel, vso, l_unkownTokens




######
# Concept-Normalization (C-Norm): Combining dynamically linear projection and shallow CNN
######
def CNorm(vst_onlyTokens, dl_terms, dl_associations, vso,
          nbEpochs=30, batchSize=64,
          l_numberOfFilters=[4000], l_filterSizes=[1],
          phraseMaxSize=15):

    # Preparing data for SLFNN and S-CNN components:
    dataSCNN, labels, l_unkownTokens, l_uncompleteExpressions = prepare2D_data(vst_onlyTokens, dl_terms, dl_associations, vso, phraseMaxSize)
    dataSLFNN = numpy.zeros((dataSCNN.shape[0], dataSCNN.shape[2]))
    for i in range( dataSCNN.shape[0]):
        numberOfToken = 0
        for embedding in dataSCNN[i]:
            if not numpy.any(embedding):
                pass
            else:
                numberOfToken += 1
                dataSLFNN[i] += embedding

        if numberOfToken > 0:
            dataSLFNN[i] = dataSLFNN[i] / numberOfToken


    # Input layers:
    inputLP = Input(shape=dataSLFNN.shape[1])
    inputCNN = Input(shape=[dataSCNN.shape[1],dataSCNN.shape[2]])


    # SLFNN component:
    ontoSpaceSize = labels.shape[2]
    denseLP = layers.Dense(units=ontoSpaceSize, use_bias=True, kernel_initializer=initializers.GlorotUniform())(inputLP)
    modelLP = Model(inputs=inputLP, outputs=denseLP)


    # Shallow-CNN component:
    l_subLayers = list()
    for i, filterSize in enumerate(l_filterSizes):

        convLayer = (layers.Conv1D(l_numberOfFilters[i], filterSize, strides=1, kernel_initializer=initializers.GlorotUniform()))(inputCNN)

        outputSize = phraseMaxSize - filterSize + 1
        pool = (layers.MaxPool1D(pool_size=outputSize))(convLayer)

        activationLayer = (layers.LeakyReLU(alpha=0.3))(pool)

        l_subLayers.append(activationLayer)

    if len(l_filterSizes) > 1:
        concatenateLayer = (layers.Concatenate(axis=-1))(l_subLayers)  # axis=-1 // concatenating on the last dimension
    else:
        concatenateLayer = l_subLayers[0]

    denseLayer = layers.Dense(ontoSpaceSize, kernel_initializer=initializers.GlorotUniform())(concatenateLayer)
    modelCNN = Model(inputs=inputCNN, outputs=denseLayer)

    convModel = Model(inputs=inputCNN, outputs=concatenateLayer)
    fullmodel = models.Sequential()
    fullmodel.add(convModel)


    # Combination of the two components:
    combinedLayer = layers.average([modelLP.output, modelCNN.output])
    fullModel = Model(inputs=[inputLP, inputCNN], outputs=combinedLayer)
    fullModel.summary()


    # Compile and train:
    fullModel.compile(optimizer=optimizers.Nadam(), loss=losses.LogCosh(), metrics=[metrics.CosineSimilarity(), metrics.MeanSquaredError()])
    fullModel.fit([dataSLFNN, dataSCNN], labels, epochs=nbEpochs, batch_size=batchSize)


    return fullModel, vso, l_unkownTokens





######
# Run class:
######
class Train(OptionParser):

    def __init__(self):

        OptionParser.__init__(self, usage='usage: %prog [options]')

        self.add_option('--word-vectors', action='store', type='string', dest='word_vectors',
                        help='path to word vectors JSON file as produced by word2vec')
        self.add_option('--word-vectors-bin', action='store', type='string', dest='word_vectors_bin',
                        help='path to word vectors binary file as produced by word2vec')
        self.add_option('--terms', action='append', type='string', dest='terms',
                        help='path to terms file in JSON format (map: id -> array of tokens)')
        self.add_option('--attributions', action='append', type='string', dest='attributions',
                        help='path to attributions file in JSON format (map: id -> array of concept ids)')
        self.add_option('--factor', action='append', type='float', dest='factors', default=[],
                        help='parent concept weight factor.')
        self.add_option('--ontology', action='store', type='string', dest='ontology',
                        help='path to ontology file in OBO format')

        self.add_option('--model', action='append', type='string', dest='model', help='path to the NN model directory')
        self.add_option('--ontology-vector', action='store', type='string', dest='ontology_vector', help='path to the ontology vector file')

        # Methods hyperparameters:
        self.add_option('--archiName', action='store', type='string', dest='archiName', help='Name of the choosen architecture')
        self.add_option('--epochs', action='store', type='int', dest='epochs', default=150, help='number of epochs (default: 150).')
        self.add_option('--batch', action='store', type='int', dest='batch', default=64, help='number of samples in batch (default: 64).')
        self.add_option('--filtersSize', action='append', type='int', dest='filtersSize', help='list of the different size of filters')
        self.add_option('--filtersNb', action='append', type='int', dest='filtersNb', help='list of the number of filters from filtersSize')
        self.add_option('--phraseMaxSize', action='store', type='int', dest='phrase_max_size', help='max considered size of phrases in inputs.')
        self.add_option('--normalizedInputs', action='store', type='string', dest='normalizedInputs', default="True", help='for each method, normalize embeddings if "True" (default: True).')


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
        if not options.terms:
            raise Exception('missing --terms')
        if not options.attributions:
            raise Exception('missing --attributions')
        if not options.model:
            raise Exception('missing --model')
        if len(options.terms) != len(options.attributions):
            raise Exception('there must be the same number of --terms and --attributions')
        if len(options.terms) != len(options.model):
            raise Exception('there must be the same number of --terms and --model')
        if len(options.factors) > len(options.terms):
            raise Exception('there must be at least as many --terms as --factor')
        if options.filtersSize is not None and options.filtersNb is not None:
            if len(options.filtersSize) != len(options.filtersNb):
                raise Exception('ERROR: number of elements in --filtersSize different from number of elements in --filtersNb')
        if options.archiName is None:
            raise Exception('missing a --archiName value ("slfnn", "scnn" or "cnorm").')
        else:
            stderr.write('Architecture name: %s \n' % options.archiName)
            stderr.flush()
            if options.archiName not in {"slfnn", "scnn", "cnorm"}:
                raise Exception('invalid --archiName value ("slfnn", "scnn" or "cnorm").')



        # Selected hyperparameters (can have an important influence...):
        if options.phrase_max_size is None:
            options.phrase_max_size = 15
            stderr.write('WARNING: no --phraseMaxSize argument. Set to %i by default \n' % options.phrase_max_size)
            stderr.flush()
        else:
            stderr.write('Selected --phrasseMaxSize argument: %i\n' % options.phrase_max_size)
            stderr.flush()
        if options.filtersNb is None:
            options.filtersNb = [3172]
            stderr.write('WARNING: no --filtersNb argument. Set to %s by default \n' % str(options.filtersNb))
            stderr.flush()
        else:
            stderr.write('Selected number of filters for each filter type: %s\n' % str(options.filtersNb))
            stderr.flush()
        if options.filtersSize is None:
            options.filtersSize = [1]
            stderr.write('WARNING: no --filtersSize argument. Set to %s by default \n' % str(options.filtersSize))
            stderr.flush()
        else:
            stderr.write('Selected sizes for each filter type: %s\n' % str(options.filtersSize))
            stderr.flush()
        epochs_number = options.epochs
        stderr.write('Selected epochs number: %i\n' % epochs_number)
        stderr.flush()
        batch_size = options.batch
        stderr.write('Selected batch size: %i\n' % batch_size)
        stderr.flush()


        # Loading ontology:
        stderr.write('loading ontology: %s\n' % options.ontology)
        stderr.flush()
        if len(options.factors) < len(options.terms):
            n = len(options.terms) - len(options.factors)
            stderr.write('defaulting %d factors to 0.6\n' % n)
            stderr.flush()
            options.factors.extend([0.6] * n)
        ontology = onto.loadOnto(options.ontology)
        first = True


        # Loading word embeddings:
        if options.word_vectors is not None:
            stderr.write('loading word embeddings: %s\n' % options.word_vectors)
            stderr.flush()
            word_vectors = loadJSON(options.word_vectors)
        elif options.word_vectors_bin is not None:
            stderr.write('loading word embeddings: %s\n' % options.word_vectors_bin)
            stderr.flush()
            EmbModel = gensim.models.Word2Vec.load(options.word_vectors_bin)
            word_vectors = dict((k, list(numpy.float_(npf32) for npf32 in EmbModel.wv[k])) for k in EmbModel.wv.vocab.keys())
        if options.normalizedInputs is not None:
            if options.normalizedInputs == "True":
                word_vectors = normalizeEmbedding(word_vectors)
                stderr.write('WARNING: Normalization of input embeddings.\n')
                stderr.flush()
        else:
            stderr.write('No normalization of input embeddings.\n')
            stderr.flush()


        # Run selected method:
        for terms_i, attributions_i, model_i, factor_i in zip(options.terms, options.attributions, options.model, options.factors):

            stderr.write('Loading terms: %s\n' % terms_i)
            stderr.flush()
            terms = loadJSON(terms_i)
            stderr.write('loading attributions: %s\n' % attributions_i)
            stderr.flush()
            attributions = loadJSON(attributions_i)

            stderr.write('factor: %f\n' % factor_i)
            stderr.flush()
            stderr.write('calculating ontological space (with factor applied)...\n')
            stderr.flush()
            vso = onto.ontoToVec(ontology, factor_i)

            stderr.write('will save trained model in: %s\n' % model_i)
            stderr.flush()


            if options.archiName == "scnn":

                l_nbFilters = options.filtersNb
                l_filterWidth = options.filtersSize
                maxPhrase = options.phrase_max_size

                model, ontology_vector, _ = SCNN(word_vectors, terms, attributions, vso,
                                             nbEpochs=epochs_number, batchSize=batch_size,
                                             l_numberOfFilters=l_nbFilters, l_filterSizes=l_filterWidth,
                                             phraseMaxSize=maxPhrase)


            elif options.archiName == "slfnn":

                model, ontology_vector, _ = SLFNN(word_vectors, terms, attributions, vso,
                                                  nbEpochs=epochs_number,
                                                  batchSize=batch_size)


            elif options.archiName == "cnorm":

                l_nbFilters = options.filtersNb
                l_filterWidth = options.filtersSize
                maxPhrase = options.phrase_max_size

                print("\nC-Norm training...")

                model, ontology_vector, _ = CNorm(word_vectors, terms, attributions, vso,
                                        nbEpochs=epochs_number, batchSize=batch_size,
                                        l_numberOfFilters=l_nbFilters, l_filterSizes=l_filterWidth,
                                        phraseMaxSize=maxPhrase)

                print("C-Norm training done.\n")

            else:
                print("ERROR: Unknown architecture: %s\n", options.archiName)
                exit(0)


            # Saving model:
            if first and options.ontology_vector is not None:
                first = False

                # translate numpy arrays into lists
                serialized_vso = dict()
                for conceptID in ontology_vector.keys():
                    serialized_vso[conceptID] = list(ontology_vector[conceptID])

                stderr.write('writing ontology vector: %s\n' % options.ontology_vector)
                stderr.flush()
                f = open(options.ontology_vector, 'w')
                json.dump(serialized_vso, f)
                f.close()

            if options.model is not None:
                stderr.write('Saving tensorflow model: %s\n' % model_i)
                stderr.flush()
                d = dirname(model_i)
                if not exists(d) and d != '':
                    makedirs(d)
                model.save(model_i)
                del model



#######################################################################################################
# Test section
#######################################################################################################

if __name__ == '__main__':

    try:

        Train().run()

    except:

        print("Test of main_train.py...\n")

        # Path to test data:
        mentionsFilePath = "../test/DATA/trainingData/terms_trainObo.json"
        attributionsFilePath = "../test/DATA/trainingData/attributions_trainObo.json"
        SSO_path = "../test/DATA/VSO_OntoBiotope_BioNLP-ST-2016.json"
        vstOnlyTokensPath = "../test/DATA/wordEmbeddings/VST_count0_size100_iter50.model"

        # Path to save test models:
        SLFNNmodelPath = "../test/DATA/learnedHyperparameters/SLFNN/"
        SCNNmodelPath = "../test/DATA/learnedHyperparameters/SCNN/"
        CNormModelPath = "../test/DATA/learnedHyperparameters/CNorm/"

        # load training data:
        print("\nLoading training data...")
        extractedMentionsFile = open(mentionsFilePath, 'r')
        dl_trainingTerms = json.load(extractedMentionsFile)
        attributionsFile = open(attributionsFilePath, 'r')
        attributions = json.load(attributionsFile)
        vst_onlyTokens = gensim.models.Word2Vec.load("../test/DATA/wordEmbeddings/VST_count0_size100_iter50.model")
        vst_onlyTokens = dict(
            (k, list(numpy.float_(npf32) for npf32 in vst_onlyTokens.wv[k])) for k in vst_onlyTokens.wv.vocab.keys())
        print("Training data loaded.\n")

        # Building of concept embeddings:
        print("Loading Semantic Space of the Ontology (SSO)...")
        SSO_file = open(SSO_path, "r")
        SSO = json.load(SSO_file)
        print("SSO loaded.\n\n")


        print("Training of all available methods...")


        print("\n\n\n\n    Training of single dense layer...\n")
        model, _, l_unknownToken = SLFNN(vst_onlyTokens, dl_trainingTerms, attributions, SSO,
                                         nbEpochs=3, batchSize=64)

        print("\n    Saving learned hyperparameters...")
        model.save(SLFNNmodelPath)
        print("    Saving done.")


        print("\n\n\n\n    Training of Shallow-CNN...\n")

        model, _, l_unknownToken = SCNN(vst_onlyTokens, dl_trainingTerms, attributions, SSO,
                                                 nbEpochs=3, batchSize=64,
                                                 l_numberOfFilters=[100], l_filterSizes=[1],
                                                 phraseMaxSize=15)

        print("\n    Saving learned hyperparameters...")
        model.save(SCNNmodelPath)
        print("    Saving done.")


        print("\n\n\n\n    Training of C-Norm...\n")
        model, _, l_unknownToken = CNorm(vst_onlyTokens, dl_trainingTerms, attributions, SSO,
                                            nbEpochs=3, batchSize=64,
                                            l_numberOfFilters=[100], l_filterSizes=[1],
                                            phraseMaxSize=15)

        print("\n    Saving learned hyperparameters...")
        model.save(CNormModelPath)
        print("    Saving done.")