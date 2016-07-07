%% USING THE AVERAGED PREDICTIONS FILE AND THE TRUE TEST-LABELS, COMPUTE ALL DOC-PIVOTED AND LABEL-PIVOTED STATISTICS 
clear;clc;close all;

%% Set up datasetlabel
files.datasetlabel = 'Yahoo_Health_Split01';

%% Set up directories for loading/saving

files.datasetdir = sprintf('../../EvaluationDatasets/%s',files.datasetlabel);           % Directory containing dataset
files.datasetfile = sprintf('%s/%s.mat', files.datasetdir,files.datasetlabel); % Dataset file

%% Set up all training/testing parameters

%=============================================
% TESTING PARAMETERS
%=============================================
% SUPER-TOPIC PARAMETERS
testparams.SUMALPHA_TOPICS = 1 ;  % TOTAL ALPHA VALUE TO DISTRIBUTE EVENLY ACROSS TOPICS (equals sum of \gamma parameter in paper)
% LABEL-DISTRIBUTION PARAMETERS
testparams.SUMALPHA_LABELS = 100; % TOTAL ALPHA VALUE TO BE TAKEN FROM THE SUPER-TOPICS (equals \eta parameter in paper)
testparams.ADDALPHA_LABELS = 1  ; % ADDITIONAL ALPHA TO DISTRIBUTE EVENLY ACROSS LABELS (equals the sum of \alpha parameter in paper)

% Number of Total test-chain samples (across all topic chains) in the saved predictions file
NSAMPLES_TESTCHAIN = 150;

%=============================================
% TRAINING PARAMETERS
%=============================================

% ---- Label-Word Training parameters ----
trainparams.PWC.ALPHA         = 50; 
trainparams.PWC.BETA          = .01;
trainparams.PWC.NCHAINS       = 5;

% ---- Topic-Label Training parameters ----
trainparams.PCT.ALPHA       = .01 ; 
trainparams.PCT.BETA        = 1 ;
trainparams.PCT.NTOPICS     = 20 ;
trainparams.PCT.NITER       = 500 ; % number of iterations for each chain

%% Set up Directory/files to load predictions from based on the training/testing parameters, and to save stats to

% Set up rootdir for all test-doc results
files.resultsdir.root = sprintf('%s/TestingOutput' , files.datasetdir); % Root of output directory (for all testoutput)
assert(exist(files.resultsdir.root,'dir')>0);
% Set up resultsdir based on all the training parameters for pCT and pWC
files.resultsdir.resultsdir = sprintf('%s/Results_TrainingParams_pWC_A%02d_B%2.3f_pCT_%02dT_%2.2fA_%2.2fB' , files.resultsdir.root, trainparams.PWC.ALPHA , trainparams.PWC.BETA , trainparams.PCT.NTOPICS , trainparams.PCT.ALPHA , trainparams.PCT.BETA);
assert(exist(files.resultsdir.resultsdir,'dir')>0)

% Set up file to load predictions from
files.predictions_totals = sprintf('%s/TotalPreds_TOPICS_SumA%02d_LABELS_SumA%02d_AddA%02d_%02dTotalTestSamples.mat',files.resultsdir.resultsdir , testparams.SUMALPHA_TOPICS , testparams.SUMALPHA_LABELS , testparams.ADDALPHA_LABELS, NSAMPLES_TESTCHAIN);
% Set up file to save all statistics to
files.statsfile_save = sprintf('%s/PredictionStats_TOPICS_SumA%02d_LABELS_SumA%02d_AddA%02d_%02dTotalTestSamples.mat',files.resultsdir.resultsdir , testparams.SUMALPHA_TOPICS , testparams.SUMALPHA_LABELS , testparams.ADDALPHA_LABELS, NSAMPLES_TESTCHAIN);

%% Load the dataset file and the predictions for all docs

load(files.datasetfile);
load(files.predictions_totals,'total_doc_preds');

%% set up prediction matrix and testlabels matrix

testlabels = full(sparse(testdata.cdidx,testdata.cidx,ones(size(testdata.cidx))));
predvals = total_doc_preds';

%%
% Get sumctrain and ndtrain
trainlabels = full(sparse(traindata.cdidx,traindata.cidx,ones(size(traindata.cidx))));
ndtrain=size(trainlabels,1);
sumctrain=sum(trainlabels); % Sumctrain gives the number of training docs for each label--used for label-pivoted binary "NPROPORTIONAL" predictions
sumdtrain = median(sum(trainlabels,2)); % Sumdtrain gives the median # training-labels / doc--used for doc-pivoted binary "NPROPORTIONAL" predictions

highval = 1;

%% Compute doc-pivoted stats
% For doc-pivoted stats use median training freq instead of sumctrain
[docstats, docfstats] = TR_Function_Compute_AllPredictionStats_New_V01(testlabels',predvals', sumdtrain, ndtrain , highval);

%% Compute label-pivoted stats
[labelstats, labelfstats] = TR_Function_Compute_AllPredictionStats_New_V01(testlabels,predvals,sumctrain,ndtrain, highval);

%% Save all statistics to a file in the resultsdirectory
save(files.statsfile_save,'docstats','docfstats','labelstats','labelfstats');