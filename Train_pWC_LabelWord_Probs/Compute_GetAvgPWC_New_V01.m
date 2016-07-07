%% Compute the averaged topic-word distributions for the LLDA model across all training chains

clear;clc;close all

%% File output (Saved to Subdirectory /TrainingOutput/pwc of the evaluation dataset):
%
%   pWC:  Word-type x Label Matrix of word-probabilities (column normalized)

%% Configure training parameters for LLDA model

trainparams.ALPHA         = 50; 
trainparams.BETA          = .01;
trainparams.NITER         = 500;
trainparams.NCHAINS       = 5;

%% Set up datasetlabel
files.datasetlabel = 'Yahoo_Health_Split01';

%% Set up directories for loading/saving

files.datasetdir = sprintf('../EvaluationDatasets/%s',files.datasetlabel);           % Directory containing dataset
files.datasetfile = sprintf('%s/%s.mat', files.datasetdir,files.datasetlabel); % Dataset file
files.resultsdir = sprintf('%s/TrainingOutput/pwc',files.datasetdir);                       % Output directory for saving trained models 
files.newclabels = sprintf('%s/TrainingOutput/newclabels.mat', files.datasetdir);           % Output file containing 'enhanced' label-strings

%% Go through all chains and load the data

nchainsloaded = 0; % Track # of chains loaded (wo that if chains are missing it won't create an error)

for chain = 1 :trainparams.NCHAINS
    %% Create the savefilename for the current chain
    load_filestr = sprintf( '%s/Topics_A%02d_B%2.3f_chain%02d_%02dIters.mat' , files.resultsdir , trainparams.ALPHA , trainparams.BETA , chain, trainparams.NITER ) ;

    %% If file exists for current chain #: load the results (WP count matrix) add it to running totals
    if exist(load_filestr, 'file')
        fprintf('Adding file for chain %02d to running total\n',chain);
        nchainsloaded = nchainsloaded+1;                % Increment count of number of chains that have been loaded
        resultsdat = load(load_filestr,'WP');           % Load the results file
        
        if nchainsloaded==1;
            totalWP = full(resultsdat.WP);              % If this if the first chain being loaded, create a full matrix for running totals
        else
            totalWP = totalWP + full(resultsdat.WP);    % Add the current counts to the running total
        end
        clear resultsdat
    end
end

%% Check that at least one results file was found, if not, print error
if nchainsloaded==0; error('No results chains were found');end

%% Create the filenames for the mat/txt files to save
trainparams.NCHAINS_COMPLETED = nchainsloaded;
files.savefile_txt = sprintf('%s/AvgPWC_A%02d_B%2.3f_%02dChains.txt', files.resultsdir, trainparams.ALPHA , trainparams.BETA , trainparams.NCHAINS_COMPLETED);
files.savefile_csv = sprintf('%s/AvgPWC_A%02d_B%2.3f_%02dChains.csv', files.resultsdir, trainparams.ALPHA , trainparams.BETA , trainparams.NCHAINS_COMPLETED);
files.savefile_mat = sprintf('%s/AvgPWC_A%02d_B%2.3f_%02dChains.mat', files.resultsdir, trainparams.ALPHA , trainparams.BETA , trainparams.NCHAINS_COMPLETED);

%% Load the training-data so we have access to wlabels
dat = load(files.datasetfile);

%% Print the averaged topic-word distributions to file

load(files.newclabels); % Load the 'enhanced' label-strings 
totalBETA = trainparams.BETA * trainparams.NCHAINS_COMPLETED; % Compute the total pseudocounts to add to each word in the topic-word matrix
WriteTopicsToCSV(files.savefile_csv, totalWP + totalBETA , dat.wlabels, newclabels );

%% Get the #topics / label in the training and test datasets

cxdtrain=sparse(dat.traindata.cidx, dat.traindata.cdidx, ones(size(dat.traindata.cidx)));
sumctrain = full(sum(cxdtrain,2));
cxdtest=sparse(dat.testdata.cidx, dat.testdata.cdidx, ones(size(dat.testdata.cidx)) , max(dat.traindata.cidx) , max(dat.testdata.cdidx)  );
sumctest = full(sum(cxdtest,2));
clearvars cxdtrain cxdtest;

%% Compute a normalized matrix of topic-word probabilities from the running totals of counts + the total pseudocounts

totalWP = totalWP + totalBETA;                          % Add the pseudocounts to the running totals of all counts
pWC=totalWP./repmat(sum(totalWP),[size(totalWP,1),1]);  % Normalize the matrix by topics to give topic-word probabilities from the counts

save(files.savefile_mat,'pWC','trainparams','sumctrain','sumctest');