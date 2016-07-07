%% Run Dependency-LDA model for multiple topic-chains and test-chains
clear;clc;close all;

%% SAMPLE TEST DOCUMENTS USING THE DEPENDENCIES MODEL:
% DEPENDENCIES ARE LEARNED BY A STANDARD LDA MODEL APPLIED TO LABELS
%
% TRAINING PARAMETERS:
%   LABEL-WORD DISTRIBUTIONS: ALPHA, BETA, NUMBER OF CHAINS
%   TOPIC-LABEL DISTRIBUTIONS: ALPHA, BETA, CHAIN NUMBER (WE CAN ONLY USE ONE CHAIN AT A TIME)--MUST INTEGRATE OVER MULTIPLE CHAINS   
%
% =========== MODEL PARAMETERS USED IN TESTING: (In 'testparams' structure variable) ===============
% SUPER-TOPIC PARAMETERS
%   - SUMALPHA_TOPICS  - Sum of prior counts distributed across all topics for each document (equals sum of \gamma parameter in paper)
% LABEL-DISTRIBUTION PARAMETERS
%   - SUMALPHA_LABELS  - Sum of prior counts computed based on predictions from the topics (equals \eta parameter in paper)
%   - ADDALPHA_LABELS  - Sum of additional smoothing counts distributed equally across all labels--equals the sum of \alpha parameter in paper
%
% PARAMETERS DETERMINING HOW WE PERFORM TEST-SAMPLING:
%
%   - SAMPLER_BURNIN    - HOW MANY ITERATIONS OF THE Z AND ZZ SAMPLER WE RUN BEFORE TAKING OUR FIRST SAMPLE
%   - SAMPLER_LAG       - HOW MANY ITERATIONS OF THE Z AND ZZ WE RUN BETWEEN TAKING ADDITIONAL SAMPLES
%   - SAMPLER_NSAMPLES  - HOW MANY SAMPLES WE TAKE FOR EACH TESTCHAIN (FOR A GIVEN CHAIN USING A SINGLE SET OF SUPER-TOPICS)
%   - NCHAINS_PER_PCT   - HOW MANY TESTCHAINS DO WE RUN FOR EACH SET OF SUPER-TOPICS LEARNED DURING TRAINING (TOPIC-LABEL SAMPLES)
%   - NCHAINS_PCT       - HOW MANY TRAINING CHAINS OF THE LABEL->TOPIC (SUPERTOPICS) WE ITERATE OVER (WE CAN RUN MULTIPLE TESTCHAINS FOR EACH SET OF SUPERTOPICS)
%
%   - ITERS_WBURNIN:    - HOW MANY ITERATIONS OF THE WORD->LABEL SAMPLER WE RUN TO INITIALIZE THE Z ASSIGNMENTS
%   - ITERS_TBURNIN:    - HOW MANY ITERATIONS OF THE LABEL->TOPIC SAMPLER WE RUN TO INITIALIZE THE ZZ ASSIGNMENTS
%   - ITERS_TOPICS      - HOW MANY ITERATIONS OF THE WORD->LABEL SAMPLER WE RUN FOR EACH ITERATION (EACH PASS OF THE Z AND ZZ SAMPLING)
%   - ITERS_TOPICS      - HOW MANY ITERATIONS OF THE LABEL->TOPIC SAMPLER WE RUN FOR EACH ITERATION (EACH PASS OF THE Z AND ZZ SAMPLING)
%
% VARIABLES USED DURING SAMPLING:
%   Z        - Assignments of words to labels
%   ZZ       - Assignments of labels to topics (z' in paper)
%   DC_Alpha - Doc x Label matrix of prior probabilies of each label for each doc (matrix of the \alpha' values in paper)
%   DT_Alpha - Doc x Topic matrix of prior probabilities of each topic for each doc (matrix of the \gamma values in paper)
%   Variables Loaded from training:
%   pWC      - Matrix of Label->Word distributions  \Phi
%   pCT      - Matrix of Topic->Label distributions \Phi' (from a single training chain)
%
%
% FOR EACH CHAIN DURING RESAMPLING:
%   
%   1 - GET INITIAL CONCEPT-ASSIGNMENTS FOR WORDS
%       a - PASS IN DOCUMENT WORDS, WITH A FLAT PRIOR ON THE CONCEPTS TOTALING (SUMALPHA_LABELS + ADDALPHA)
%       b - AFTER WORDBURNIN ITERATIONS, WE RETURN VECTOR Z CONTAINING ASSIGNMENTS OF WORDS-LABELS FOR THE DOCUMENT 
%
%   2 - GET TOPIC-DISTRIBUTIONS FOR ALL DOCUMENTS (BY MAKING TOPIC ASSIGNMENTS FOR ALL WORD-LABEL ASSIGNMENTS)
%       a - PASS IN THE Z-ASSIGNMENTS TO THE TOPIC SAMPLER **AS IF THEY WERE WORDS*** : 
%               [DPT] = GibbsSampler_TagLDA_DCAlpha_V03(Z, wdidx, DTAlpha, pCT, NITER, SEED, 2); % DTAlpha SHOULD BE A FLAT PRIOR ON SUPER-TOPICS (PERHAPS WEIGHTED BY WORD PROBABILITIES)
%       b - USE THE FIXED SUPER-TOPIC PROBABILITIES FROM THE SINGLE SUPER-TOPIC CHAIN
%       c - 

%% Set up datasetlabel
files.datasetlabel = 'Yahoo_Health_Split01';

%% Set up directories for loading/saving

files.datasetdir = sprintf('../EvaluationDatasets/%s',files.datasetlabel);           % Directory containing dataset
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
% DEPENDENCY SAMPLER SAMPLING PARAMETERS
%=============================================
ITERS_WBURNIN   = 2;    % NUMBER OF ITERATIONS TO INITIALLY SAMPLE WORDS
ITERS_TBURNIN   = 2;    % NUMBER OF ITERATIONS TO INITIALLY SAMPLE TOPICS (FOR THE INITIAL BURNED-IN WORD ASSIGNMENTS)

ITERS_LABELS    =  1;   % HOW MANY ITERATIONS WE RUN FOR THE LABEL UPDATES (SAMPLING Z ASSIGNMENTS FOR WORDS)
ITERS_TOPICS    =  1;   % HOW MANY ITERATIONS WE RUN FOR THE TOPIC UPDATES (SAMPLING ZZ ASSIGNMENTS FOR EACH Z-ASSIGNMENT)

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

% Make sure the averaged Label-Word file exists
files.pwc.resultsfile =  sprintf('%s/AvgPWC_A%02d_B%2.3f_%02dCHAINS.mat' ,  files.pwc.loaddir, trainparams.PWC.ALPHA, trainparams.PWC.BETA, trainparams.PWC.NCHAINS);
assert(exist(files.pwc.resultsfile,'file')~=0);

%% Set up Directory to save results to, based on the training parameters

% Set up rootdir for all output
files.outputdir.root = sprintf('%s/TestingOutput' , files.datasetdir); % Root of output directory (for all testoutput)
if ~exist(files.outputdir.root,'dir');mkdir(files.outputdir.root);end
% Set up savedir based on all the training parameters
files.outputdir.savedir = sprintf('%s/Results_TrainingParams_pWC_A%02d_B%2.3f_pCT_%02dT_%2.2fA_%2.2fB' , files.outputdir.root, trainparams.PWC.ALPHA , trainparams.PWC.BETA , trainparams.PCT.NTOPICS , trainparams.PCT.ALPHA , trainparams.PCT.BETA);
if ~exist(files.outputdir.savedir,'dir' ); mkdir(files.outputdir.savedir );end

%% LOAD THE DATASET FILE AND THE AVERAGED P(W|C) FILE

% Load the data for the test-documents
load(files.datasetfile,'testdata');

% Load the averaged label-word distributions
load(files.pwc.resultsfile,'pWC');

%% GET BASIC TESTDATA INFO AND INITIALIZE PRIOR MATRICES

nd = max(testdata.wdidx);
nt = trainparams.PCT.NTOPICS;
nc = size(pWC,2);

% CREATE THE DT_Alpha MATRIX (Matrix of prior probabilities of each supertopic given each document). This doesn't change during sampling
DT_Alpha = repmat( testparams.SUMALPHA_TOPICS / nt , nd , nt);

%%
%-------------------------------------------------------------------------------------
%  ITERATE OVER THE TOPIC CHAINS (make this a parfor loop if using parallel toolbox)
% ------------------------------------------------------------------------------------
for pctchain = 1 : NCHAINS_PCT
    %% Load the Topic-> Label distributions \Phi' for the current topic-chain 
    
    % Load the trained topic->label count matrix from file
    pct_filestr = sprintf('%s/%02dTopics_A%2.3f_B%2.3f_chain%02d_%02dIters.mat', files.pct.loaddir , trainparams.PCT.NTOPICS , trainparams.PCT.ALPHA , trainparams.PCT.BETA , pctchain , trainparams.PCT.NITER ) ;
    pctdata = load(pct_filestr,'WP');
    % Normalize count matrix to compute Topic->Label probs, \Phi'
    pCT = pctdata.WP + trainparams.PCT.BETA ;
    pCT = pCT ./ repmat( sum(pCT,1), [nc, 1]);
    clear pctdata
    
    %% Set up the root of the filenames to use for saving all samples for current topic-chain
    filenm_root =sprintf( '%s/TOPICS_SumA%02d_LABELS_SumA%02d_AddA%02d_PCTChain%02d' , files.outputdir.savedir , testparams.SUMALPHA_TOPICS , testparams.SUMALPHA_LABELS , testparams.ADDALPHA_LABELS , pctchain );
    
    %-----------------------------------------------------------
    %  ITERATE OVER MULTIPLE CHAINS PER TOPIC CHAIN (PUT PARFOR HERE)
    % ----------------------------------------------------------
    for testchain = 1 : NCHAINS_PER_PCT
        %% Initialize Some Variables
        % Document x Label matrix of priors on labels
        DC_Alpha  = repmat( (testparams.SUMALPHA_LABELS+testparams.ADDALPHA_LABELS) / nc , nd , nc);
        
        %% Check whether results file for first sample already exists, skip it if it does
        filenm_testchain = sprintf('%s_%02d_Sample%02d_Iter%02d.mat' , filenm_root , testchain , 1 , SAMPLER_BURNIN) ;
        fileflag = exist(filenm_testchain,'file');
        
        %% IF FIRST FILE DOESN'T EXIST, RUN CHAIN
        if ~fileflag
            %% Set up random seed, print info about current chain to command-window
            SEED = testchain; 
            fprintf('\n--- Topic-Chain %02d; Test-Chain %02d: Collecting Samples At Iterations %s --- \n', pctchain, testchain, num2str( SAMPLER_BURNIN : SAMPLER_LAG : SAMPLER_BURNIN+(SAMPLER_NSAMPLES-1)*SAMPLER_LAG) );
            
            % INITIAL BURN-IN OF WORD->LABEL ASSIGNMENTS
            fprintf('Running Initialization of Word->Label (Z) Assignments...\n' );tic;
            [DP Z] = GibbsSampler_TagLDA_DCAlpha_V04(testdata.widx, testdata.wdidx, DC_Alpha, pWC, ITERS_WBURNIN , SEED, OUTPUT); DP=[];
            SEED = SEED+1;
            
            % INITIAL BURN-IN OF LABEL->TOPIC ASSIGNMENTS
            fprintf('Running Initialization of Label->Topic (ZZ) Assignments...\n' )
            [DPT ZZ] = GibbsSampler_TagLDA_DCAlpha_V04( Z, testdata.wdidx, DT_Alpha, pCT, ITERS_TBURNIN, SEED, OUTPUT);
            etime = toc; fprintf( 'Elapsed time for Z and ZZ initializations: %3.1f secs\n' , etime );
            SEED = SEED+1;
            
            % Track the total number of sampling iterations (post initialization)
            n_iters_total = 0;
            
            %% TAKE MULTIPLE SAMPLES - FOR FIRST SAMPLE ITERATE OVER WORDS / LABELS "SAMPLER_BURNIN" TIMES, OTHERWISE ITERATE "SAMPLER_LAG" TIMES
            for SAMPLE = 1 : SAMPLER_NSAMPLES
             
                % Determine Number of Iterations to run. 
                %       First sample:  SAMPLER_BURNIN iterations
                %       Subsequent samples: SAMPLER_LAG iterations
                if SAMPLE == 1
                    NITER = SAMPLER_BURNIN ; % FOR FIRST SAMPLE OF CHAIN, WE RUN "SAMPLER_BURNIN" PASSES THROUGH BOTH SAMPLERS
                else
                    NITER = SAMPLER_LAG;
                end
                
                % For all iterations until saving a sample, run the ZZ sampler and then the Z sampler
                for iter = 1 : NITER
                    
                    % ============== Sample Word-Assignments (Z) starting at their previous assignments, using updated DC_Alpha Matrix ========== %
                    % If SAMPLE is 1 ITER is 1, then we don't need to run the ZZ sampler (since we just ran the initialization)
                    if ~( SAMPLE == 1 && iter == 1)
                        fprintf( 'Topic-Chain %02d; Test-Chain %02d; Sample %02d; Iter %02d - Running ZZ Sampler (Label->Topic assignments)\n' , pctchain , testchain , SAMPLE , n_iters_total);
                        [DPT ZZ] = GibbsSampler_TagLDA_DCAlpha_V04( Z, testdata.wdidx, DT_Alpha, pCT, ITERS_TOPICS, SEED, OUTPUT, ZZ);
                        SEED = SEED+1;
                    end
                    
                    % -------- Compute prior matrix DC_Alpha from the DPT matrix of Label->Topic (ZZ) Assignments ------- 
                    
                    % FIRST CREATE A NORMALIZED DOCUMENT X LABEL PROBABILITY MATRIX
                    normDPT = DPT + DT_Alpha';  % ADD THE TOPIC PRIORS BACK INTO DPT
                    normDPT = normDPT ./ repmat( sum(normDPT) , nt , 1) ; % NORMALIZE THE DPT MATRIX SO THAT dPT SUMS TO 1 FOR EACH TOPIC
                    
                    % NOW MULTIPLY P(TOPIC|DOCS) AND P(LABELS|TOPICS) TO COMPUTE PRIOR P(LABELS|DOCUMENTS)
                    DC_Alpha = pCT * normDPT ;  % MATRIX MULTIPLICATION CREATES A [LABEL X DOC] MATRIX FROM [DOC x TOPIC] AND [TOPIC x LABEL] MATRICES
                    DC_Alpha = DC_Alpha ./ repmat( sum(DC_Alpha) , nc , 1) ; % NORMALIZE THE [LABEL X DOC PROBABILITY MATRIX]
                    DC_Alpha = DC_Alpha * testparams.SUMALPHA_LABELS ; % MULTIPLY BY SUMALPHA_LABELS TO TURN INTO COUNTS (Where sum of each doc's alpha vector equals SUMALPHA_LABELS)
                    DC_Alpha = DC_Alpha + testparams.ADDALPHA_LABELS / nc; % ADD THE FLAT COUNTS BACK IN
                    DC_Alpha = DC_Alpha' ;      % TRANSPOSE TO MAKE [DOC X LABEL] DIMENSIONALITY
                    
                    % ============== Sample Word-Assignments (Z) starting at their previous assignments, using updated DC_Alpha Matrix ========== %
                    fprintf('Topic-Chain %02d; Test-Chain %02d; Sample %02d; Iter %02d - Running Z Sampler (Word->Label assignments) \n' , pctchain , testchain , SAMPLE , n_iters_total);
                    [DPC Z] = GibbsSampler_TagLDA_DCAlpha_V04(testdata.widx, testdata.wdidx, DC_Alpha, pWC , ITERS_LABELS, SEED, OUTPUT , Z);
                    etime2 = toc;
                    SEED = SEED+1;
                    
                    % Number of iterations in total 
                    n_iters_total = n_iters_total + 1 ;
                    fprintf( 'Iter %02d Complete. Elapsed time for sampling: %3.1f secs. Approximately %3.1f secs / iteration\n' , n_iters_total, etime2, (etime2 - etime) / n_iters_total );
                end
                
                SAMPLER_ITER = SAMPLER_BURNIN + (SAMPLE-1) * SAMPLER_LAG ;
                assert(SAMPLER_ITER==n_iters_total);
                
                % WRITE FILES AND SAVE VARIABLES
                fprintf( 'Topic-Chain %02d; Test-Chain %02d; Sample %02d; Iter %02d - Saving Sample\n' , pctchain , testchain , SAMPLE , n_iters_total);
                filenm_testchain = sprintf('%s_%02d_Sample%02d_Iter%02d.mat' , filenm_root , testchain , SAMPLE , SAMPLER_ITER) ;
                savevars_DPC_DPT( filenm_testchain , DPC , DPT );
                
            end  % END FOR SAMPLER = 1: SAMPLER_NSAMPLES
            
            % AFTER EACH CHAIN IS COMPLETE, SAVE THE FINAL Z STATES
            fprintf( 'Saving final state (Z and ZZ assignments) at sample %02d for: Topic-Chain %02d; Test-Chain %02d; Iter %02d \n' , SAMPLER_ITER , pctchain , testchain ,  n_iters_total);
            filenm_testchainfinal = sprintf('%s_%02d_FinalState_Sample%02d_Iter%02d.mat' , filenm_root , testchain , SAMPLE , SAMPLER_ITER) ;
            savevars_Z_ZZ_FinalStates( filenm_testchainfinal , Z , ZZ);
            
        else
            fprintf( '\nSkipping testchain %d\n' , testchain );
        end
    end

end

return