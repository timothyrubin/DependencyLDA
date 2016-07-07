%% Train multiple chains of the Labeled-LDA model

clear; clc; close all
%% File output (Saved to Subdirectory /TrainingOutput/pwc of the evaluation dataset):
%   DP: Document  x Label Count Matrix
%   WP: Word-Type x Label Count Matrix

%% Model & Sampling Parameters 
%
% We use standard LDA terminology in code, which doesn't align with paper.
% Mapping from Paper parameters to variables in code are as follows:
%   trainparams.ALPHA = \eta
%   trainparams.BETA  = \beta_w
%   Z = z
%   DP = \Theta (Doc->Label Probabilities)
%   WP = \Phi   (Label->Word Probabilities)
%   cxdmat: This is a binary label-x-document matrix of indicators: if label c assigned to document d, cxd(c,d)=1

trainparams.ALPHA   = 50;   % Sum of prior-counts on topics (\eta)
trainparams.BETA    = .01;  % Individual prior count on each word (\beta_w)
trainparams.NITER   = 500;  % Number of iterations per chain during training
trainparams.NCHAINS = 5;    % Number of training chains (which we will average over for computing a final estimate)
trainparams.OUTPUT  = 2;    % Controls verbosity of compiled sampler
trainparams.PrintTopics = 1;% Print the label-word distributions to a file?

%% Set up datasetlabel
files.datasetlabel = 'Yahoo_Health_Split01';

%% Set up directories for loading/saving

files.datasetdir = sprintf('../EvaluationDatasets/%s',files.datasetlabel);           % Directory containing dataset
files.datasetfile = sprintf('%s/%s.mat', files.datasetdir,files.datasetlabel); % Dataset file
files.savedir = sprintf('%s/TrainingOutput/pwc',files.datasetdir);         % Output directory for saving trained models 
if ~exist(files.savedir); mkdir(files.savedir) ; end                        % If it doesn't exist creeate output directory
files.newclabels = sprintf('%s/TrainingOutput/newclabels.mat' , files.datasetdir);        % File to save 'enhanced' label-strings for the label-types
fprintf('Training Label-Word Distributions For Dataset in directory: \t%s\n', files.datasetdir);

%% Load the relevant dataset variables from the dataset file
load(files.datasetfile,'traindata','clabels','wlabels');

%% Create and save an updated array of strings for label-types, which appends the # Docs/Label to each label-string
cxdmat = sparse( traindata.cidx , traindata.cdidx , ones( size( traindata.cidx )));
docsperlabel = full( sum( cxdmat , 2 ));
newclabels = cell( 1,  length(clabels));
for t=1:length(newclabels)
   newclabels{t} = sprintf( '%s (D=%d)' , clabels{ t } , docsperlabel( t )); 
end
if ~exist(files.newclabels); save(files.newclabels,'newclabels'); end

%% Run gibbs sampler for all chains
for chain = 1 : trainparams.NCHAINS
    %% Create filename-strings for the current chain
    filenm_root=sprintf( '%s/Topics_A%02d_B%2.3f_chain%02d_%02dIters' , files.savedir , trainparams.ALPHA , trainparams.BETA , chain, trainparams.NITER ) ;
    filenm_txt = sprintf('%s.txt' , filenm_root) ;
    filenm_csv = sprintf('%s.csv' , filenm_root) ;
    filenm_mat = sprintf('%s.mat' , filenm_root) ;    
    %% If file doesn't exist, run the chain
    if ~exist(filenm_mat,'file');
        % Run Gibbs sampler
        fprintf( '\nRunning chain %d\n' , chain ); 
        tic;
        SEED = chain;
        % GibbsSamplerLDATAGS3 uses the corrected smoothing of the alpha parameterization
        [ WP,DP,Z ] = GibbsSamplerLDATAGS3( traindata.widx , traindata.wdidx , cxdmat , trainparams.NITER , trainparams.ALPHA , trainparams.BETA , SEED , trainparams.OUTPUT );
        
        DP = DP' * 1.0; % Transpose DP and Multiply by 1 to ensure we remove zero entries

        %% Print info about sampling progress time to command window
        etime = toc;
        fprintf( 'Elapsed time for sampling: %3.3f secs\n' , etime );
        fprintf( 'Time per iteration: %3.3f secs\n' , etime / trainparams.NITER );

        %% Write files and save variables
        savevars(filenm_mat , DP , WP );
        if trainparams.PrintTopics
            WriteTopicsToCSV(filenm_csv, WP + trainparams.BETA , wlabels, newclabels )
        end
    else % If file already exists skip it
        fprintf( 'Skipping chain %02d (Already exists)\n' , chain );
    end
end
fprintf('Sampling complete for all chains training label-word distributions: p(w|c) \n')