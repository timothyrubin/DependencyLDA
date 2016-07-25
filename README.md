# DependencyLDA

This is a MATLAB (and C) implementation of the Dependency-LDA, Prior-LDA and Flat-LDA models presented in the following paper:

Rubin, T. N., Chambers, A., Smyth, P., & Steyvers, M. (2012). [Statistical topic models for multi-label document classification. ](http://arxiv.org/abs/1107.2462) Machine learning, 88(1-2), 157-208.
________________________________________________________
# Overview

Dependency-LDA, Prior-LDA and Flat-LDA are methods for multi-label document classification, where each document is associated with one or more labels from the set of all labels. 

Training of models involves learning: 
- label->word distributions for all labels
- topic->label distributions. For Dependency-LDA we learn T of these distributions, and for Prior-LDA we learn a single distribution, corresponding to each label's frequency in the training set.

At test time, we learn the assignments of test documents' word-tokens to labels, and (for Dependency-LDA) the assignments of label-tokens to topics.

##### Note on variable naming

Note that the variable in the code is not equivalent in some cases what is used in the paper. The code is thoroughly commented, and the comments describe the correspondences between variable names and model parameters from the paper.

_________________________________________________
# Setup

The code should run out-of-the-box if you are on a Macintosh. If you are not on a mac, you will need to re-compile all of the .mex code (from .c source code) and put the samplers in the corresponding directories.


##### Compiling the Mex files from C

All matlab scripts call compiled c code (which matlab compiles into .mex files) when sampling the z/z' indicators ). I have included code that was compiled on a Mac in the appropriate directories. If you are on a mac, the .mex files should work as is. 

If the you are not on a mac or the code doesn't work for you, you should recompile all of the c code in the "/Mex_Samplers" directory by running the following script within that directory: 

		compileCode64bit.m

Note that matlab sometimes gives an error (depending on your computer's) when compiling "GibbsSamplerLDA_v2.c". The issue that this causes has been addressed in the script that uses this code ("Train_pCT_New_V01.m"), so this should not cause any problems.



##### Current code configurations

All code is currently configured to run on the 'Yahoo_Health_Split01' evaluation dataset included in the package. 

For training, the code is currently configured to Train the following:
- word->label distributions (\phi) used for all models (Dependency-LDA, Prior-LDA and Flat-LDA)
- label->topic distributions (\phi') for use with the Dependency-LDA model. Parameters for the label->topic distributions will need to be changed to train the Prior-LDA model.

For testing, the code is currently configured to run both:
- Dependency-LDA
- Prior-LDA model. 

Changing a couple of the Prior-LDA parameters will allow testing of the Flat-LDA model.

_________________________________________________
# Inference for Training and Testing of models

The steps required to train the model, and then make predictions using Dependency-LDA are as follows:

### Training:

1. Train the label->word distributions:
 * Run "ldatagmodel_TrainPWC_V01.m" in the "/Train_pWC_LabelWord_Probs" directory. This samples z for training documents and is used to train \Phi, which gives: p(w|c).
 * Run "Compute_GetAvgPWC_New_V01.m" to infer \Phi for multiple training chains
 * Run "Compute_GetAvgPWC.m" to compute an average \Phi over all training chains
			
2. Train the topic->label distributions
 *  Run "Train_pCT_New_V01.m" in the "/Train_pCT_TopicLabel_Probs" directory. This samples z' for training documents and is used to train \Phi', which gives: p(c|t).
  
** Note **  The topic->label distributions cannot be averaged over, since topics are not going to be aligned across chains. Instead, separate test-chains are run for each set of topics computed during training (and the test-chains can then be averaged over)
- For Dependency-LDA, any setting of NTOPICS>1 can be used (see paper for our recommended settings), and we sample multiple chains
- For Prior-LDA and Flat-LDA, train a single chain with NTOPICS=1;  Prior-LDA is a special case of Dependency-LDA, in which there is a single Topic->Label distribution, where that topic's distribution over labels is proportional to each label's training frequency. Flat-LDA is a special case of Prior-LDA, in which no weight is put on the \eta parameter when computing each document's \alpha prior distribution over labels.

### Testing (Dependency-LDA):

- Run the "DependencyLDA_Sample_TestDocuments.m" script in "/Test_DependencyLDA_TestDocuments".
 * For each training chain of \Phi', this will run test-chain(s) that sample z and z' for test documents, based on the averaged \Phi. The predictions for test-chains will later be averaged over.

### Testing (Prior-LDA):

- Run the "PriorLDA_Sample_TestDocuments.m" script in "Test_PriorLDA_TestDocuments".  
 * Prior-LDA uses a single training chain of \Phi', in which NTOPICS = 1

### Testing (Flat-LDA):

- Run the "PriorLDA_Sample_TestDocuments.m" script in "Test_PriorLDA_TestDocuments".  
 * Since Flat-LDA is a special case of Prior-LDA, we can use the same code to make predictions with Flat-LDA, and only need to modify a couple of parameters.
 * Flat-LDA uses a single training chain of \Phi', in which NTOPICS = 1 and \eta = 1 (the \eta parameter from the paper is named "SUMALPHA_LABELS" in the code)
 * For additional details see "Note on Prior-LDA and Flat-LDA" further down in this document

### Final Predictions, and evaluation of predictions (all models):

Compute final predictions for test-documents and then compute statistics:
 * Code for computing averaged predictions across all test-chains for all test-documents, and then evaluating these predictions are in the subdirectories:
 * **Dependency-LDA**: "/Test_DependencyLDA_TestDocuments/Compute_TotalDP_Predictions_New_V05"
 * **Prior-LDA**:		"/Test_PriorLDA_TestDocuments/Compute_TotalDP_Predictions_New_V05"
 
 Within each of the above subdirectories, do the following two steps:
	
 1. Run the following script to average predictions on test-documents, across chains: 

 	TestDocs_Compute_TotalDP_TotalPriors_AllChains_Samples_New_V05.m

 2. Run the following script to Compute all evaluation statistics for test-document predictions: 
 
 		Compute_AllStats_Predictions.m

The file containing all computed statistics for both document-pivoted and label-pivoted predictions will be saved in the appropriate subdirectory of the evaluation dataset.

Document-Pivoted statistics are stored in variable "docstats" and label-pivoted statistics are in variable "labelstats". Note that many more statistics are computed than were actually presented in our paper.
_________________________________________________
### Note on Prior-LDA and Flat-LDA

As noted above, Prior-LDA can be treated a special case of Dependency-LDA in which there is a single Topic / Topic->Label distribution, where the topic's distribution over labels is proportional to the training frequency. Since all documents z/c variables can only be sampled from this one topic, document's \alpha' priors over labels end up being proportional to this one topic's distributions over labels.

Flat-LDA can be treated as a special case of this Prior-LDA model, where the vector \alpha' is computed by putting no weight on the output from the topic->label part of the model. This can be achieved using the Prior-LDA code, by setting the \eta parameter ("SUMALPHA_LABELS" in the code) to ZERO. Therefore only the \alpha ("ADDALPHA_LABELS" in the code) is used in computing \alpha' for each document, which gives a flat prior over labels. For example, as described in the paper, for the Yahoo Datasets we used SUMALPHA_LABELS=0 and ADDALPHA_LABELS=100. 

_________________________________________________
### Results examples for all models (Dependency-LDA, Prior-LDA, Flat-LDA)

Examples of completed results and test-document statistics for all three models on the "Yahoo_Health_Split01" dataset are provided in the subdirectories of the Yahoo_Health_Split01 dataset. You can use these as a guide for using the provided code to run all three models.

The .mat files containing all evaluation statistics for both doc-pivoted and label-pivoted predictions for the three models are here:

- Dependency-LDA:
	EvaluationDatasets/Yahoo_Health_Split01/TestingOutput/Results_TrainingParams_pWC_A50_B0.010_pCT_20T_0.01A_1.00B/PredictionStats_TOPICS_SumA01_LABELS_SumA100_AddA01_150TotalTestSamples.mat

- Prior-LDA:
	EvaluationDatasets/Yahoo_Health_Split01/TestingOutput/Results_TrainingParams_pWC_A50_B0.010_pCT_01T_0.01A_10.00B/PredictionStats_TOPICS_SumA01_LABELS_SumA70_AddA30_150TotalTestSamples.mat

- Flat-LDA:
	EvaluationDatasets/Yahoo_Health_Split01/TestingOutput/Results_TrainingParams_pWC_A50_B0.010_pCT_01T_0.01A_10.00B/PredictionStats_TOPICS_SumA01_LABELS_SumA00_AddA100_150TotalTestSamples.mat
_________________________________________________
### Modifying the code parameters and running on different datasets

All code is currently configured to run the Yahoo_Health_Split1 dataset, using parameters described in the paper (except taking fewer samples). To run with a different dataset, the following changes will need to be made in each of the scripts:

**Changing dataset-label:**
- Change the "files.datasetlabel" to the corresponding evaluation dataset

**Changing parameters:**
- Change the Training Parameters as appropriate. For training \Phi and \Phi' the training parameters all are put into a "trainparams" structure variable near the top of the scripts. E.g., "trainparams.BETA" in the "Train_pCT_New_V01.m" file corresponds to the \beta_c parameter in our paper.
- Change the Testing Parameters as appropriate (in the DependencyLDA_Sample_TestDocuments.m and PriorLDA_Sample_TestDocuments.m scripts). These parameters are in the cell "TEST PARAMETERS" in the matlab script. Note again that the variable names are different from parameter names in the paper, but the relationship is documented in the code.

Please see the paper for our general ecommendations regarding setting parameters for a new dataset.
________________________________________________________

# DATASET FORMATTING:

All datasets within the directory "Evaluation_Datasets" should work if the "files.datasetlabel" variable is changed as appropriate. Included in this directory are the following datasets used in our paper:
	NYT
	Yahoo_Arts (5 separate train/test splits)
	Yahoo_Health (5 separate train/test splits)

Please follow the variable-naming conventions used in these datasets if you want to run the code on additional datasets that are not included. Here is a summary of how the datasets are configured:

##### Each dataset contains two structure variables:

- "traindata" contained indices for training documents.
- "testdata" contained indices for test-documents.


##### Within the structure variables (e.g., "traindata") we would have the following fields:

word-tokens:
	widx = vector of word-token indices
	wdidx = vector of word-document indices
label-tokens:
	cidx = vector of label-token indices
	cdidx = vector of label-document indices

##### Additionally, each dataset should contain the following two variables:

wlabels = cell-array of the strings corresponding to token-indices
clabels = cell-array of the strings corresponding to label-indices

These are used for printing out "topics" for visual inspection (e.g. using the "WriteTopicsToCSV" function).

If your dataset does not have strings for the "wlabels" or "clabels", you can just use insert generic strings (e.g., "word1", "word2" and "label1" "label2"). I have done this for the Yahoo datasets, which were sent to us without any label/word strings.

________________________________________________________
### ADDITIONAL COMMENTS:

For Dependency-LDA, the code is not fully optimized (and can sometimes be slow for large datasets). Specifically, while the compiled code is fast (for sampling the z and z' variables) it repeatedly passes variables back and forth between matlab and the compiled code. 

If you would like to make the code run faster, you can try modifying the sampling parameters that control how often the data is passed between matlab and the compiled code:
- ITERS_LABELS
- ITERS_TOPICS 

which control how many full iterations to run on z and z' before the updated values are passed back to matlab). Alternatively (ideally), all of the code would be compiled, but we do not have an implementation of that.

