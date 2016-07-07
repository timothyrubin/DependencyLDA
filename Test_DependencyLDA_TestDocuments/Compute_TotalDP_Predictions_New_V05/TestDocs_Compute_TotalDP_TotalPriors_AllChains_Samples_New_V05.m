%% AVERAGE PREDICTION TOTALS OVER MULTIPLE TEST-CHAINS FOR THE DEPENDENCY-LDA MODEL
clear;clc;close all;

%% Set up datasetlabel
files.datasetlabel = 'Yahoo_Health_Split01';

%% Set up directories for loading/saving

files.datasetdir = sprintf('../../EvaluationDatasets/%s',files.datasetlabel);           % Directory containing dataset
files.datasetfile = sprintf('%s/%s.mat', files.datasetdir,files.datasetlabel); % Dataset file
files.pwc.loaddir = sprintf('%s/TrainingOutput/pwc',files.datasetdir);          % Load directory for trained label-word distributions
files.pct.loaddir = sprintf('%s/TrainingOutput/pct',files.datasetdir);          % Load directory for trained topic-label distributions
assert(exist(files.pwc.loaddir,'dir') & exist(files.pct.loaddir,'dir'));        % Assert that the directories exist
files.newclabels = sprintf('%s/TrainingOutput/newclabels.mat' , files.datasetdir);        % File to save 'enhanced' label-strings for the label-types

%% TEST PARAMETERS

%=============================================
% HYPER-PARAMETERS
%=============================================
% SUPER-TOPIC PARAMETERS
testparams.SUMALPHA_TOPICS = 1 ;  % TOTAL ALPHA VALUE TO DISTRIBUTE EVENLY ACROSS TOPICS (equals sum of \gamma parameter in paper)
% LABEL-DISTRIBUTION PARAMETERS
testparams.SUMALPHA_LABELS = 100; % TOTAL ALPHA VALUE TO BE TAKEN FROM THE SUPER-TOPICS (equals \eta parameter in paper)
testparams.ADDALPHA_LABELS = 1  ; % ADDITIONAL ALPHA TO DISTRIBUTE EVENLY ACROSS LABELS (equals the sum of \alpha parameter in paper)

%=============================================
% FULL SAMPLER SAMPLING PARAMETERS
%=============================================
SAMPLER_BURNIN   = 50 ; % HOW MANY ITERATIONS OF THE FULL SAMPLER (LABEL UPDATES AND TOPIC UPDATES) WE RUN BEFORE TAKING OUR FIRST SAMPLE
SAMPLER_LAG      = 5  ; % HOW MANY ITERATIONS OF THE FULL SAMPLER WE RUN BETWEEN TAKING ADDITIONAL SAMPLES
SAMPLER_NSAMPLES = 15  ; % HOW MANY SAMPLES WE TAKE FOR EACH CHAIN (FOR A GIVEN CHAIN USING A SINGLE SET OF SUPER-TOPICS)

NCHAINS_PCT      = 5  ; % HOW MANY SETS OF SUPER-TOPIC CHAINS WE ITERATE OVER
NCHAINS_PER_PCT  = 2  ; % HOW MANY TEST CHAINS WE RUN FOR EACH SET SET OF SUPER-TOPICS

OUTPUT = 0;
%% SET UP ALL TRAINING-PARAMETERS (USED FOR TRAINING LABEL->WORD AND TOPIC->LABEL DISTRIBUTIONS)

% ---- Label-Word Training parameters ----
trainparams.PWC.ALPHA         = 50; 
trainparams.PWC.BETA          = .01;
trainparams.PWC.NCHAINS       = 5;

% ---- Topic-Label Training parameters ----
trainparams.PCT.ALPHA       = .01 ; 
trainparams.PCT.BETA        = 1 ;
trainparams.PCT.NTOPICS     = 20 ;
trainparams.PCT.NITER       = 500 ; % number of iterations for each chain

%% Set up pWC file directory and make sure it exists
files.pwc.resultsfile =  sprintf('%s/AvgPWC_A%02d_B%2.3f_%02dCHAINS.mat' ,  files.pwc.loaddir, trainparams.PWC.ALPHA, trainparams.PWC.BETA, trainparams.PWC.NCHAINS);
assert(exist(files.pwc.resultsfile,'file')~=0);

%% Set up Directory to load test-documents results from, based on the training/testing parameters

% Set up rootdir for all test-doc results
files.resultsdir.root = sprintf('%s/TestingOutput' , files.datasetdir); % Root of output directory (for all testoutput)
assert(exist(files.resultsdir.root,'dir')>0);
% Set up resultsdir based on all the training parameters for pCT and pWC
files.resultsdir.resultsdir = sprintf('%s/Results_TrainingParams_pWC_A%02d_B%2.3f_pCT_%02dT_%2.2fA_%2.2fB' , files.resultsdir.root, trainparams.PWC.ALPHA , trainparams.PWC.BETA , trainparams.PCT.NTOPICS , trainparams.PCT.ALPHA , trainparams.PCT.BETA);
assert(exist(files.resultsdir.resultsdir,'dir')>0)



%% LOAD THE DATASET FILE AND THE AVERAGED P(W|C) FILE

% Load the data for the test-documents
load(files.datasetfile,'testdata');

% Load the averaged label-word distributions
load(files.pwc.resultsfile,'pWC');


%% GET BASIC TESTDATA INFO AND INITIALIZE PRIOR MATRICES

% nd = length(testdata.docid);
%nd = max(testData.testdata.wdidx);
nd = max(testdata.wdidx);
nt = trainparams.PCT.NTOPICS;
nc = size(pWC,2);

%% Set up variables tracking how many test-samples are being included in predictions

ntestchainsamples = 0;
ntestchains = 0;
npctchains = 0;
ntotalsamples_pctchain = 0;

%% Create matrices for storing total_DC_Alpha and total_DC_Counts

total_DC_Counts = zeros(nc, nd); % Count matrix storing document counts of all Z assignments of words->labels across all chains
total_DC_Alpha = zeros(nc, nd); % Prior matrix storing priors over all labels for each document

% DT_Alpha MATRIX 
%       Matrix of prior probabilities of each supertopic given each document). This is used in computing the alpha prior for each
%       document given the document->topic probs and topic->label probs for the topic-chain
DT_Alpha = repmat( testparams.SUMALPHA_TOPICS / nt , nd , nt);

%%
%-----------------------------------------------------------
%  ITERATE OVER THE TOPIC CHAINS
% ----------------------------------------------------------
for pctchain = 1 : NCHAINS_PCT
    fprintf('=======================================================================\n')
    fprintf('\t  GOING THROUGH ALL TEST-CHAINS FOR TOPIC-CHAIN %02d \n' , pctchain) ;
    fprintf('=======================================================================\n')
    
    % LOAD THE PCT INFO FOR THIS PCT CHAIN
    pct_filestr = sprintf('%s/%02dTopics_A%2.3f_B%2.3f_chain%02d_%02dIters.mat', files.pct.loaddir , trainparams.PCT.NTOPICS , trainparams.PCT.ALPHA , trainparams.PCT.BETA , pctchain , trainparams.PCT.NITER ) ;
    pctdata = load(pct_filestr,'WP');
    
    % Normalize count matrix to compute Topic->Label probs, \Phi'
    pCT = pctdata.WP + trainparams.PCT.BETA ;
    pCT = pCT ./ repmat( sum(pCT,1), [nc, 1]);
    clear pctdata
    
    %% Set up the root of the filenames to use for saving all samples for current topic-chain
    filenm_root =sprintf( '%s/TOPICS_SumA%02d_LABELS_SumA%02d_AddA%02d_PCTChain%02d' , files.resultsdir.resultsdir , testparams.SUMALPHA_TOPICS , testparams.SUMALPHA_LABELS , testparams.ADDALPHA_LABELS , pctchain );
    
    %-----------------------------------------------------------
    %  ITERATE OVER MULTIPLE TEST-CHAINS PER TOPIC CHAIN 
    % ----------------------------------------------------------
    for testchain = 1 : NCHAINS_PER_PCT
        %% FOR LOADING FILES -- FOR NOW JUST CHECK TO MAKE SURE THAT ALL CHAINS HAVE BEEN COMPLETED

        % Compute the final sample iteration if test-chain was completed
        SAMPLE = SAMPLER_NSAMPLES; 
        SAMPLER_ITER = SAMPLER_BURNIN + (SAMPLE-1) * SAMPLER_LAG;
        
        % Set up filename for the test-chain and see if it was completed
        filenm_testchain = sprintf('%s_%02d_Sample%02d_Iter%02d.mat' , filenm_root , testchain , SAMPLE , SAMPLER_ITER) ;
        fileflag = logical(exist(filenm_testchain));
        
        %% ONLY GO THROUGH ALL SAMPLES FOR THE TEST-CHAIN IF LAST CHAIN HAS BEEN COMPLETED
        if fileflag
            fprintf('\n--- Adding samples from: Topic-Chain %02d; Test-Chain %02d: --- \n', pctchain, testchain);
            
            % % MATRIXES TO AD ADD UP COUNTS FOR ALL SAMPLES WITHIN ONE TEST-CHAIN
            % totalDPT_testchain = zeros(nt,nd);
            % totalDPC_testchain = zeros(nc,nd);
            
            %% ============== ITERATE OVER ALL SAMPLES FOR A SINGLE TEST-CHAIN=========
            for SAMPLE = 1 : SAMPLER_NSAMPLES
             
                SAMPLER_ITER = SAMPLER_BURNIN + (SAMPLE-1) * SAMPLER_LAG ;
                fprintf( 'Loading Test-Chain: Topic-Chain %02d; Test-Chain %02d; Sample %02d; Iter %02d \n' , pctchain , testchain , SAMPLE , SAMPLER_ITER);
                filenm_testchain = sprintf('%s_%02d_Sample%02d_Iter%02d.mat' , filenm_root , testchain , SAMPLE , SAMPLER_ITER) ;
                chaindata = load(filenm_testchain,'DPC','DPT');
                
                %% Compute the Prior Matrix from the Doc->Topic probs
                % FIRST CREATE A NORMALIZED DOCUMENT X LABEL PROBABILITY MATRIX
                normDPT = chaindata.DPT + DT_Alpha';  % ADD THE TOPIC PRIORS BACK INTO DPT
                normDPT = normDPT ./ repmat( sum(normDPT) , nt , 1) ; % NORMALIZE THE DPT MATRIX SO THAT dPT SUMS TO 1 FOR EACH TOPIC
                
                % NOW MULTIPLY P(TOPIC|DOCS) AND P(LABELS|TOPICS) TO COMPUTE PRIOR P(LABELS|DOCUMENTS)
                DC_Alpha = pCT * normDPT ;  % MATRIX MULTIPLICATION CREATES A [LABEL X DOC] MATRIX FROM [DOC x TOPIC] AND [TOPIC x LABEL] MATRICES
                DC_Alpha = DC_Alpha ./ repmat( sum(DC_Alpha) , nc , 1) ; % NORMALIZE THE [LABEL X DOC PROBABILITY MATRIX]
                DC_Alpha = DC_Alpha * testparams.SUMALPHA_LABELS ; % MULTIPLY BY SUMALPHA_LABELS TO TURN INTO COUNTS (Where sum of each doc's alpha vector equals SUMALPHA_LABELS)
                DC_Alpha = DC_Alpha + testparams.ADDALPHA_LABELS / nc; % ADD THE FLAT COUNTS BACK IN
                %DC_Alpha = DC_Alpha' ;      % TRANSPOSE TO MAKE [DOC X LABEL] DIMENSIONALITY

                %% Add the DC_Alpha and DC_Counts to the running totals
                
                total_DC_Alpha = total_DC_Alpha + DC_Alpha;
                total_DC_Counts = total_DC_Counts + chaindata.DPC; 
                
                
                %totalDPT_testchain = totalDPT_testchain + chaindata.DPT ;
                %totalDPC_testchain = totalDPC_testchain + chaindata.DPC ;
                
                ntestchainsamples = ntestchainsamples +1 ;  % Tracks Number of Samples overall
                ntotalsamples_pctchain = ntotalsamples_pctchain +1; % Tracks number of test samples for each pCT chain
                
            end %_______"for SAMPLE=1:SAMPLER_NSAMPLES"_________%
            
            % ADD THE COUNT TOTALS FOR ALL SAMPLES FROM THIS TESTCHAIN TO THE RUNNING TOTAL ACROSS ALL TESTCHAINS FOR PCT
            %totalDPT_pctchain = totalDPT_pctchain + totalDPT_testchain;
            %totalDPC_pctchain = totalDPC_pctchain + totalDPC_testchain;
            ntestchains = ntestchains + 1; % INCREASE THE TOTAL NUMBER OF TESTCHAINS FOR THIS PCT CHAIN

        else;  fprintf( '*** SKIPPING CHAIN:   PCT CHAIN %02d, TEST CHAIN %02d.  (NOT ALL TESTCHAINS COMPLETED YET) ***\n' , pctchain , testchain);
        
        end %_______"if fileflag"_________%

    end  %_______" testchain = 1:NCHAINS_PER_PCT "_________%  
    
    %%  HERE WE HAVE TO STORE ALL THE PRIOR COUNTS FOR TOPIC-ASSIGNMENTS ( ZZ / DPT ) BECAUSE THEY *DO NOT* AVERAGE OVER TOPIC CHAINS
    
    %     % SAVE RESULTS FOR THIS TOPIC CHAIN, ONLY IF THE TOPIC HAS DATA
    %     if ntotalsamples_pctchain>0
    %         npctchains = npctchains + 1;
    %         %pctresults(pctchain).pct_testc =  pct_testc  ;
    %         pctresults(pctchain).pct_testc =  pCT ;
    %         pctresults(pctchain).totalDPT  =  totalDPT_pctchain ;
    %         pctresults(pctchain).totalDPC  =  totalDPC_pctchain ;
    %         pctresults(pctchain).ntotalsamples_pctchain = ntotalsamples_pctchain ;
    %     end
    
    %% SOME HOUSECLEARNING OF VARIABLES
    clearvars pCT 
    %ntotalsamples_pctchain = 0; % Reset This count after storing results
end

%% Create a single combined prediction:
% Following the paper, for each document we set the total count of priors equal to each document's total in the word->label count matrix
% (can do this by normalizing the word->label count matrix and prior matrix separately, then adding them and renormalizing)
total_DC_Alpha = total_DC_Alpha./repmat(sum(total_DC_Alpha,1),size(total_DC_Alpha,1),1);
total_DC_Counts = total_DC_Counts./repmat(sum(total_DC_Counts,1),size(total_DC_Counts,1),1);
total_doc_preds = total_DC_Alpha + total_DC_Counts;

% Normalize the total_doc_preds
total_doc_preds = total_doc_preds./repmat(sum(total_doc_preds,1), size(total_doc_preds,1), 1);

%% Save the resulting output to the results directory
files.savefilenm = sprintf('%s/TotalPreds_TOPICS_SumA%02d_LABELS_SumA%02d_AddA%02d_%02dTotalTestSamples.mat',files.resultsdir.resultsdir , testparams.SUMALPHA_TOPICS , testparams.SUMALPHA_LABELS , testparams.ADDALPHA_LABELS, ntestchainsamples);
save(files.savefilenm,'total_doc_preds','total_DC_Alpha','total_DC_Counts');

return