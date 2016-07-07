%% Run Standard LDA on a document x label matrix (Gives Topic->Label Distributions phi')

clear; clc; close all

%% File output (Saved to Subdirectory /TrainingOutput/pct of the evaluation dataset):
%   DP: Document  x Topic Count Matrix
%   WP: Label-Type x Topic Count Matrix

%% Model & Sampling Parameters 
%
% We use standard LDA terminology in code, which doesn't align with paper.
% Mapping from Paper parameters to variables in code are as follows:
%   trainparams.ALPHA   = \gamma
%   trainparams.BETA    = \beta_c
%   trainparams.NTOPICS = T.  Note that for Prior-LDA should run 1 chain with T=1
%   Z = z'
%   DP = \Theta'
%   WP = \Phi'

trainparams.ALPHA       = .01 ; 
trainparams.BETA        = 1 ;
trainparams.NTOPICS     = 20 ;
trainparams.NITER       = 500 ; % number of iterations for each chain
trainparams.NCHAINS     = 5 ;
trainparams.OUTPUT      = 2;        % Controls verbosity of compiled sampler
trainparams.PrintTopics = 1;    % Print the topic-label distributions to a file?

%% Set up datasetlabel
files.datasetlabel = 'Yahoo_Health_Split01';

%% Set up directories for loading/saving

files.datasetdir = sprintf('../EvaluationDatasets/%s',files.datasetlabel);           % Directory containing dataset
files.datasetfile = sprintf('%s/%s.mat', files.datasetdir,files.datasetlabel); % Dataset file
files.savedir = sprintf('%s/TrainingOutput/pct',files.datasetdir);              % Output directory for saving trained models 
if ~exist(files.savedir); mkdir(files.savedir) ; end                            % If it doesn't exist create output directory
files.newclabels = sprintf('%s/TrainingOutput/newclabels.mat' , files.datasetdir);   % File to load 'enhanced' label-strings for the label-types
fprintf('\n Training Topic-Label Distributions For Dataset in directory: \t%s\n', files.datasetdir);

%% Load training data from file
load(files.datasetfile,'traindata'); % Load the traindata (containing the label-indices)
if ~exist(files.newclabels,'file')
    error('** newclabels file does not exist: train the label-word distributions first');
else
    load(files.newclabels);
end

%% Run gibbs sampler for all chains
for chain = 1 : trainparams.NCHAINS
    %% Create filename-strings for the current chain
    filenm_root= sprintf( '%s/%02dTopics_A%2.3f_B%2.3f_chain%02d_%02dIters' , files.savedir , trainparams.NTOPICS , trainparams.ALPHA , trainparams.BETA , chain , trainparams.NITER ) ;
    filenm_txt = sprintf('%s.txt' , filenm_root) ;
    filenm_csv = sprintf('%s.csv' , filenm_root) ;
    filenm_mat = sprintf('%s.mat' , filenm_root) ;
    %% IF FILE DOESN'T EXIST, RUN CHAIN
    if ~exist(filenm_mat,'file');
        % Run Gibbs sampler
        fprintf( '\nRunning chain %d\n' , chain );
        tic;
        SEED = chain;
        [ WP,DP,Z ] = GibbsSamplerLDA_v2( traindata.cidx , traindata.cdidx , trainparams.NTOPICS , trainparams.NITER , trainparams.ALPHA , trainparams.BETA , SEED , trainparams.OUTPUT );
        
        %% Modify output from c-code in case needed
        % Reconstruct WP/DP from the vector of Z's (to deal with potential problems compilation errors for the sampler)
        WP = full(sparse(traindata.cidx,Z,ones(size(traindata.cidx))));
        DP = full(sparse(traindata.cdidx,Z,ones(size(traindata.cdidx))));
        
        %% Print info about sampling progress to command window
        etime = toc;
        fprintf( 'Elapsed time for sampling: %3.3f secs\n' , etime );
        fprintf( 'Time per iteration: %3.3f secs\n' , etime / trainparams.NITER );
        
        %% Write files and save variables
        savevars(filenm_mat , DP , WP);
        if trainparams.PrintTopics
            WriteTopicsToCSV(filenm_csv, WP + trainparams.BETA , newclabels )
        end
    else % If file already exists skip it
        fprintf( 'Skipping chain %02d (Already exists)\n' , chain );
    end
end
fprintf('Sampling complete for all chains training topic-label distributions\n')