**# C-Norm (Concept NORMalization)**


The method implemented in this repository enables to reproduce equivalent results than those published in:<br />
C-Norm: a Neural Approach to Few-Shot Entity Normalization<br />
Arnaud Ferré-1, Louise Deléger-1, Robert Bossy-1, Pierre Zweigenbaum-2, Claire Nédellec-1<br />
1-Université Paris-Saclay, INRAE MaIAGE, Jouy-en-Josas, France<br />
2-Université Paris-Saclay, CNRS LIMSI, Orsay, France<br />


Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


**Command lines:**

C-Norm training on tool dataset:
`python module_train/main_train.py --word-vectors-bin test/DATA/wordEmbeddings/VST_count0_size100_iter50.model --ontology test/DATA/OntoBiotope_BioNLP-ST-2016.obo --terms test/DATA/trainingData/terms_train
.json --factor 0.6 --model test/DATA/learnedHyperparameters/CNorm/ --archiName cnorm --epochs 3 --batch 64 --attributions test/DATA/trainingData/attributions_train.json`

C-Norm prediction on tool dataset (on dev set):<br/>
`python module_predictor/main_predictor.py --word-vectors-bi test/DATA/wordEmbeddings/VST_count0_size100_iter50.model --ontology test/DATA/OntoBiotope_BioNLP-ST-2016.obo --terms test/DATA/trainingData/term
s_dev.json --factor 0.6 --model test/DATA/learnedHyperparameters/CNorm/ --output test/DATA/predictedData/CNorm_pred.txt`


**Demo run:**
You can directly run the two scripts: module_train/main_train.py and the module_predictor/main_predictor.py.