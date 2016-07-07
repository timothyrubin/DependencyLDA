function [stats, fstats] = TR_Function_Compute_AllPredictionStats_New_V01(testlabels,predvals,sumctrain,ndtrain, highval)
%% In v03: Differentiate between if we're doing document-pivoted or label pivoted.
% I think most elegant way to do this is to pass in sumctrain as a 1xC vector for
% label-pivoted and a scalar (equal to median # documents) for doc-pivoted
%% FUNCTION HELP
%
% Function can be used to compute all Document-Pivoted or Label-Pivoted
% Statistics. It detects the pivot based on input 'sumctrain'. I'll
% describe the inputs it expects for both scenarios here. Note that
% sumctrain and ndtrain are relevant only for the binary 'trainprop'
% predictions, where the number of positive predictions is set proportional
% to the average training frequencies
%
%   [stats, fstats] = TR_Function_Compute_AllPredictionStats_New_V01(testlabels,predvals,sumctrain,ndtrain, highval)
%   
%  --- Inputs for Label-Pivoted Statistics ---
%
%       - testlabels: a Document x Label (DxC) matrix of True label values, where:
%               0 = negative instance
%               1 = positive instance
%       - predvals: a Document x Labels (DxC) matrix of real-valued predictions,
%       - sumctrain: A 1xC Vector of counts of the number of times each label occured in the training data (used for the "TRAIN_PROP" F1 evaluations)
%       - ndtrain:  A scalar giving the total number of documents in the training dataset
%       - Highval  Scalar indicating the directionality of 'predvals'. If
%               highval>0. Higher values are predicted more likely to be positive labels
%               highval <=0 Higher values are predicted less likely to be positive labels
%   
%  --- Inputs for Document-Pivoted Statistics ---
%
%       - testlabels: a Label x Document(CxD) matrix of True label values, where:
%               0 = negative instance
%               1 = positive instance
%       - predvals: a Label x Document (CxD) matrix of real-valued predictions,
%       - sumctrain: A Scalar Of the median number of labels per training document
%       - ndtrain:  A scalar giving the total number of documents in the training dataset (not used)
%       - Highval  Scalar indicating the directionality of 'predvals'. If

%% BASED ON WHAT THE "highval" IS, CONVERT VALUES TO RANKINGS
%
if highval>0
    [dummy predranks] = sort(predvals,'descend');
else
    [dummy predranks] = sort(predvals,'ascend');
end

%% Determine if we are operating on doc-pivoted or label-pivoted (only affects a couple thing)
if length(sumctrain)==1
    pivot = 'd';
else
    pivot = 'c';
end

%% COMPUTE SOME BASIC PARAMETER VALUES

testlabels = sparse(testlabels);

[nd nc] = size(testlabels);

nc1 = sum(testlabels) ; % NUMBER OF POSITIVE INSTANCES FOR EACH LABEL
nc0 = nd - nc1 ;        % NUMBER OF NEGATIVE INSTANCES FOR EACH LABEL

testc = find(nc1) ;     % INDEX OF ALL LABELS THAT HAVE AT LEAST 1 POSITIVE EXAMPLE

%% BUILD STRUCTURE FOR SAVING ALL STATS

stats.TPR = zeros(nd,nc) ;
stats.FPR = zeros(nd,nc) ;

stats.Recall_Interp = zeros(nd+1,nc) ;
stats.Prec_Interp = zeros(nd+1,nc) ;

counts.TP = zeros(nd,nc);
counts.FP = zeros(nd,nc);
counts.FN = zeros(nd,nc);
counts.TN = zeros(nd,nc);

stats.nctrain = zeros(1,nc); % NUMBER OF TRAINING-DOCS ASSIGNED LABEL-C
stats.nctest = zeros(1,nc);  % NUMBER OF TEST-DOCS ASSIGNED LABEL-C

stats.AUC_PR = zeros(1,nc); % AUC FOR EACH LABEL
stats.AUC_ROC = zeros(1,nc); % AUC FOR EACH LABEL


%% RANKING STATISTICS (MULTI-LABEL RANKING STATS)

rankstats.OneError = zeros(1,nc);
rankstats.IsError = zeros(1,nc);
rankstats.Margin = zeros(1,nc);
rankstats.Coverage = zeros(1,nc);
rankstats.RankLoss = zeros(1,nc);

%% BREAK-EVEN LOCATION INDEX (USED LATER TO AGGREGATE COUNTS AT SPECIFIC LOCATION)

BEP.BEP_Loc = zeros(1,nc);

%% ======== SET UP VALUES FOR F1 & RELATED ======
stats.optF1 = zeros(1,nc);
stats.AvgPrec = zeros(1,nc); % AUC FOR EACH LABEL

ErrorTR = zeros(nc,1); % TEMPORARY WAY (FAST) TO COMPUTE AUC

%% SET UP VALUES FOR MANN-WHITNEY U

stats.MannU_Z = zeros(1,nc);
stats.MannU_AUC = zeros(1,nc);
stats.MannU_P = zeros(1,nc);

%% SET UP VALUES USED FOR AUC COMPUTATIONS

rnkvals = (1 : nd)' ;   % A VECTOR OF ALL VALUES 1 - ND

xroc = zeros(nd+2,1);
yroc = zeros(nd+2,1);
xroc(end) = 1;
yroc(end) = 1;

total_xroc = zeros(nd+2,1);
total_yroc = zeros(nd+2,1);
total_xroc(end) = 1;
total_yroc(end) = 1;

%% SAVE SOME BASIC VALUES
stats.testc = testc;    % SAVE INDEX OF ALL TESTED LABELS

%% LOOP OVER ALL LABELS THAT EXIST IN THE TEST-SET
for c = testc
    %% GET THE RANKING INFORMATION FOR THE DOCUMENTS THAT WERE (TRULY) ASSIGNED LABEL-C
    
    n1 = nc1(c) ;        % THE NUMBER OF DOCS ASSIGNED THE LABEL
    n0 = nd-n1 ;         % THE NUMBER OF DOCS NOT ASSIGNED THE LABEL
    
    cdranks = predranks(:,c) ;      % RANKING OF DOCUMENTS FOR LABEL C
    
    cd1 = find(testlabels(:,c)) ;   % DOCS THAT WERE ASSIGNED LABEL C
    r1 = ismember(cdranks,cd1) ;    % LOGICAL INDEX OF ALL RANKINGS OF TRUE LABELS
    
    crnk1 =  find(r1) ;             % THE PREDICTED RANKINGS OF ALL DOCUMENTS ASSIGNED LABEL C
    
    %% FOR ALL RANKINGS COMPUTE THE TOTAL TP, TN, FP, FN
    
    rnkTP = cumsum(r1);             % FOR ALL POSSIBLE THRESHOLDS, THE NUMBER OF TRUE POSITIVES     * TRUE POSITIVES *
    rnkFP = rnkvals-rnkTP;          % FOR ALL POSSIBLE THRESHOLDS, THE NUMBER OF FALSE POSITIVES    * FALSE ALARMS *
    rnkFN = n1 - rnkTP;             % FOR ALL POSSIBLE THRESHOLDS, THE NUMBER OF FALSE NEGATIVES    * MISSES *
    rnkTN = n0 - rnkFP;             % FOR ALL POSSIBLE THRESHOLDS, THE NUMBER OF TRUE NEGATIVES     * CORRECT REJECTIONS *
    
    %% SAVE THE COUNTS (USEFUL FOR MACRO AND MICRO-AVERAGING)
    
    counts.TP(:,c) = rnkTP;
    counts.FP(:,c) = rnkFP;
    counts.FN(:,c) = rnkFN;
    counts.TN(:,c) = rnkTN;
    
    %% COMPUTE THE TPRATE & FP-RATE, AND USIE THIS TO COMPUTE THE ROC CURVE
    
    rnk_TPR = rnkTP ./ (rnkTP + rnkFN); % EQUIV TO RECALL , &  I.E., SENSITIVITY   
    rnk_FPR = rnkFP ./ (rnkFP + rnkTN); % SPECIFICITY = 1-FPR

    xroc(2:1+nd) = rnk_FPR;
    yroc(2:1+nd) = rnk_TPR;
    
    stats.AUC_ROC(c) = 1-trapz(yroc,xroc); % COMPUTE THE AUC USING THE TRAPEZOIDAL APPROXIMATION OF THE CURVE
    
    % AUC:  ALTERNATIVE AUC COMPUTATION METHOD (RAPID)
    S0 = sum(crnk1);
    ErrorTR(c) = 1 - (S0 - (n1^2 + n1)/2) / (n1 * n0);

    %% SAVE ALL INFORMATION FOR THIS LABEL
    
    stats.TPR(:,c) = rnk_TPR;
    stats.FPR(:,c) = rnk_FPR;
   
    stats.nctest(c) = n1;
    % If sumctrain is a scalar, we are operating on documents, and therefore sumctrain is meaningless
    if isequal(pivot,'c')
        stats.nctrain(c) = sumctrain(c);
    end
    
    %% COMPUTE THE PRECISION AND RECALL AT DIFFERENT RANKINGS
    
    rnkPREC = rnkTP ./ (rnkTP + rnkFP) ;
    rnkRECALL = rnkTP ./ (rnkTP + rnkFN) ;
   
    %%  COMPUTE AUC FOR PRECISION-RECALL 
    % =====================================================================
    % ================ COMPUTE AUC-PR:  USE METHOD TO MATCH 2006 ICML PAPER on ROC and Precision Recall curves 
    
    % ADD A POINT CORRESPONDING TO RECALL = 0, AND A CORRESPONDING LOCATION (PRECISION VALUE WILL BE INTERPOLATED AFTER)
    rnkRECALL_INTERP = [0; rnkRECALL] ; % ADD A ZERO-POINT
    rnkPREC_INTERP = [1; rnkPREC] ;  % ADD 1 POINT-(BY DEFINITION) 
    
    % JUST INTERPOLATE UP TO THE FIRST RECALLED ITEM (TO COMPUTE AREA-UNDER PR-CURVE AS THEY DO IN ICML PAPER)
    rnkPREC_INTERP(1:crnk1(1)) = rnkPREC(crnk1(1));
    
    stats.AUC_PR(c) = trapz(rnkRECALL_INTERP,rnkPREC_INTERP) ;    % COMPUTE THE AREA

    stats.Recall_Interp(:,c) = rnkRECALL_INTERP;
    stats.Prec_Interp(:,c) = rnkPREC_INTERP;

    %% COMPUTE THE F1-SCORE ACROSS ALL RANK-VALS AND THE OPTIMAL-F1 SCORES
    rnkF1 = 2.*(rnkRECALL .* rnkPREC) ./ (rnkRECALL + rnkPREC) ;
    [stats.optF1(c) BEP.BEP_Loc(c) ] = max(rnkF1);  % GET THE BREAK-EVEN POINT (WE CAN COMPUTE SOME MACROS AT THIS SPECIFIC POINT
    
    %% COMPUTE MEAN AVERAGE PRECISION
    stats.AvgPrec(c) = mean(rnkPREC(crnk1));
    
    %% Matthews correlation coefficient
    
    rnk_MCC = ((rnkTP.*rnkTN)-(rnkFP.*rnkFN)) ./ sqrt((rnkTP+rnkFP).*(rnkTP+rnkFN).*(rnkTN+rnkFP).*(rnkTN+rnkFN));
    
    %% MANN-WHITNEY U: COMPUTE STATS BASED ON ACTUAL VALUES (NON-PARAMETRIC)
    
    rnksum1 = sum(crnk1) ; % SUM OF RANKS FOR TRUE LABELS
    MannU1 =  rnksum1 - (n1*(n1+1))/2 ; % THE MANN-WHITNEY U VALUE
    
    approx_meanU1 = n1*n0 / 2 ;
    approx_stdU1 = sqrt((n1*n0)*(n1+n0+1) / 12) ;
    approx_Z = - ( (MannU1 - approx_meanU1) / approx_stdU1 ) ;
    
    stats.MannU_Z(c) = approx_Z;
    stats.MannU_AUC(c) = 1-MannU1/(n1*n0) ;
    
    
    %     %% GET MANN-WHITNEY VALS USING STATS TOOLBOX
    vals_true = predvals( find(testlabels(:,c)) , c);
    vals_false = predvals( ~testlabels(:,c) , c);
    [mannu_p,mannu_h,mannu_stat] = ranksum( vals_true, vals_false);
    
    stats.MannU_P(c) = mannu_p;
    
    %% COMPUTE THE MULTI-LABEL RANKSTATS
    
    firsterr = find(~ismember(1:n1+1, crnk1),1,'first');
    
    npairstotal = n1*n0 ; 
    npairsbad = sum(crnk1 - [1:n1]') ; % RANK-LOSS IS EQUIVALENT TO ONE-MINUS THE MANN-WHITNEY U TEST-STATISTIC
    pctmisordered = npairsbad/npairstotal ; % (* SAME VALUE AS VARIABLE "MannU1" )

    rankstats.OneError(c) = ~ismember(1,crnk1);     % One-Error:  Is the top prediction an error?
    rankstats.IsError(c) = ~(max(crnk1)==n1);       % IsError:  Is there an error in the order?
    rankstats.Margin(c) = crnk1(end) - firsterr;
    rankstats.Coverage(c) = max(crnk1);
    rankstats.RankLoss(c) = pctmisordered; % Exactly the same as:  1-stats.MannU_AUC
    

end


%% COMPUTE MACRO-MACRO-AVERAGES OF THE AUC VALUES

stats.macro_AUC_PR = mean(stats.AUC_PR(testc)) ;
stats.macro_AUC_ROC = mean(stats.AUC_ROC(testc));
stats.macro_AvgPrec = mean(stats.AvgPrec(testc));

stats.macro_OptF1 = mean(stats.optF1(testc)) ;
stats.macro_MannU_Z = mean(stats.MannU_Z(testc)) ;
stats.macro_MannU_P = mean(stats.MannU_P(testc)) ;


%% COMPUTE (MACRO-AVG) RANKING-STATS

stats.rankstats.OneError = 100*mean(rankstats.OneError(testc));
stats.rankstats.IsError = 100*mean(rankstats.IsError(testc));
stats.rankstats.AvgPrec = 100*mean(stats.AvgPrec(testc));
stats.rankstats.RankLoss = 100*mean(rankstats.RankLoss(testc)); % EQUIVALENT TO 100*(1-AUC_ROC)
stats.rankstats.Margin = mean(rankstats.Margin(testc));
stats.rankstats.Coverage = mean(rankstats.Coverage(testc)) - 1;

% 100*(1-stats.macro_AUC_ROC)-stats.rankstats.RankLoss % THE SAME

%% COMPUTE THE MICRO-AVERAGES OF THE AUC VALUES (OLD METHOD--NOT CORRECT)
% - THIS METHOD IS INCORRECT BECAUSE WE ARE ADDING UP COUNTS (WE ARE IN RANK-VALUE SPACE--NEED TO STAY IN ROC SPACE OR PR-SPACE
% ========= FIRST COMPUTE THE TOTAL COUNTS OF 
totalTP = sum(counts.TP,2) ;
totalFP = sum(counts.FP,2) ;
totalFN = sum(counts.FN,2) ;
totalTN = sum(counts.TN,2) ;

% ========= MICRO-AVG FOR ROC-CURVE
total_TPR = totalTP ./ (totalTP + totalFN);
total_FPR = totalFP ./ (totalFP + totalTN);

total_xroc(2:1+nd) = total_FPR;
total_yroc(2:1+nd) = total_TPR;

total_AUC = 1-trapz(total_yroc,total_xroc);

stats.micro_AUC_ROC = total_AUC;


%% * AT CALIBRATID LOCATION (POSPRED = TRUEPOS) POINT (FOR EACH LABEL) * :  GO OVER EACH TEST-LABEL, AND COMPUTE MICRO-STATS BASED ON BREAK-EVEN POINT LOCATIONS
sumctest = sum(testlabels);

% COMPUTE MICRO-AVERAGE OF F1-VALUES AT EACH OF THE CALIB-POINTS (WHERE TRUE POS = POSIIVE PREDICTIONS)
CALIBP.TP = zeros(nc,1);
CALIBP.FP = zeros(nc,1);
CALIBP.TN = zeros(nc,1);
CALIBP.FN = zeros(nc,1);

for c = testc
    
    loc = sumctest(c);

    CALIBP.TP(c) = counts.TP(loc,c);
    CALIBP.FP(c) = counts.FP(loc,c);
    CALIBP.TN(c) = counts.TN(loc,c);
    CALIBP.FN(c) = counts.FN(loc,c);

end

% ===== GET THE MICRO-AVG AND MACRO-AVG OF LABEL PERFORMANCE WHEN CALIBRATED TO TRUE #   ====
CALIBP.sumTP = sum(CALIBP.TP) ;
CALIBP.sumFP = sum(CALIBP.FP) ;
CALIBP.sumTN = sum(CALIBP.TN) ;
CALIBP.sumFN = sum(CALIBP.FN) ;

CALIBP.FPR = CALIBP.FP ./ (CALIBP.FP + CALIBP.TN) ;
CALIBP.TPR = CALIBP.TP ./ (CALIBP.TP + CALIBP.FN) ; 
CALIBP.REC = CALIBP.TP ./ (CALIBP.TP + CALIBP.FN) ;
CALIBP.PREC = CALIBP.TP ./ (CALIBP.TP + CALIBP.FP) ;


% ===== GET A MICRO-AVERAGE (FIRST TOTAL THE COUNTS)==== %
CALIBP.total_precision = CALIBP.sumTP ./ (CALIBP.sumTP + CALIBP.sumFP) ;
CALIBP.total_recall = CALIBP.sumTP ./ (CALIBP.sumTP + CALIBP.sumFN) ;
% COMPUTE THE MICRO-F1 ACROSS ALL LOCATIONS
CALIBP.total_F1 = 2.*(CALIBP.total_recall .* CALIBP.total_precision) ./ (CALIBP.total_recall + CALIBP.total_precision) ;
CALIBP.microF1 = CALIBP.total_F1;

% ===== GET A MACRO-AVERAGE (FIRST COMPUTE F1 THEN TAKE AVG)==== %
CALIBP.F1=2.*(CALIBP.REC .* CALIBP.PREC) ./ (CALIBP.REC + CALIBP.PREC) ; % COMPUTE EACH F1 
CALIBP.F1(isnan(CALIBP.F1') & sumctest>0) = 0; % F1 VALUES THAT ARE ISNAN = 0
CALIBP.macroF1 = mean(CALIBP.F1(testc));
CALIBP.macroRECALL = mean(CALIBP.REC(testc));
CALIBP.macroPREC = mean(CALIBP.PREC(testc));

% ===== SAVE F1, PRECISION, AND RECALL, IN STATS
stats.Calib_microF1 = CALIBP.microF1;                   % MICRO-F1
stats.Calib_macroF1 = CALIBP.macroF1;                   % MACRO-F1

stats.Calib_microPrecision = CALIBP.total_precision ;   % MICRO-PRECISION
stats.Calib_macroPrecision = CALIBP.macroPREC;        % MACRO-PRECISION

stats.Calib_microRecall = CALIBP.total_recall;          % MICRO-RECALL
stats.Calib_macroRecall = CALIBP.macroRECALL;               % MACRO-RECALL

%% *AT TRAINING-PROPORTIONALITY POINT (PROPORTIONAL TO # TRAINING LABELS)

% If doc-pivoted, we use median training frequency (which is sumdtrain)
if isequal(pivot,'d')
    TRAINPROP.loc = repmat(sumctrain,nc,1);
% If label-pivoted, we use: sumctrain(label)/ndtrain * ndtest
else 
    TRAINPROP.prob=zeros(nc,1);
    TRAINPROP.prob(testc) = sumctrain(testc)/ndtrain;
    TRAINPROP.loc = ceil(TRAINPROP.prob * nd);
end
    
%{
% pctrain=sumctrain./ndtrain;
TRAINPROP.prob=zeros(nc,1);
TRAINPROP.prob(testc) = sumctrain(testc)/ndtrain;
%TRAINPROP.loc = TRAINPROP.prob;
TRAINPROP.loc = ceil(TRAINPROP.prob * nd);
%}
% COMPUTE MICRO-AVERAGE OF F1-VALUES AT EACH OF THE CALIB-POINTS (WHERE TRUE POS = POSIIVE PREDICTIONS)
TRAINPROP.TP = zeros(nc,1);
TRAINPROP.FP = zeros(nc,1);
TRAINPROP.TN = zeros(nc,1);
TRAINPROP.FN = zeros(nc,1);

for c = testc
    
    loc = TRAINPROP.loc(c);

    TRAINPROP.TP(c) = counts.TP(loc,c);
    TRAINPROP.FP(c) = counts.FP(loc,c);
    TRAINPROP.TN(c) = counts.TN(loc,c);
    TRAINPROP.FN(c) = counts.FN(loc,c);

end

% ===== GET THE MICRO-AVG AND MACRO-AVG OF LABEL PERFORMANCE WHEN CALIBRATED TO TRUE #   ====
TRAINPROP.sumTP = sum(TRAINPROP.TP) ;
TRAINPROP.sumFP = sum(TRAINPROP.FP) ;
TRAINPROP.sumTN = sum(TRAINPROP.TN) ;
TRAINPROP.sumFN = sum(TRAINPROP.FN) ;

TRAINPROP.FPR = TRAINPROP.FP ./ (TRAINPROP.FP + TRAINPROP.TN) ;
TRAINPROP.TPR = TRAINPROP.TP ./ (TRAINPROP.TP + TRAINPROP.FN) ; 
TRAINPROP.REC = TRAINPROP.TP ./ (TRAINPROP.TP + TRAINPROP.FN) ;
TRAINPROP.PREC = TRAINPROP.TP ./ (TRAINPROP.TP + TRAINPROP.FP) ;


% ===== GET A MICRO-AVERAGE (FIRST TOTAL THE COUNTS)==== %
TRAINPROP.total_precision = TRAINPROP.sumTP ./ (TRAINPROP.sumTP + TRAINPROP.sumFP) ;
TRAINPROP.total_recall = TRAINPROP.sumTP ./ (TRAINPROP.sumTP + TRAINPROP.sumFN) ;
% COMPUTE THE MICRO-F1 ACROSS ALL LOCATIONS
TRAINPROP.total_F1 = 2.*(TRAINPROP.total_recall .* TRAINPROP.total_precision) ./ (TRAINPROP.total_recall + TRAINPROP.total_precision) ;
TRAINPROP.microF1 = TRAINPROP.total_F1;

% ===== GET A MACRO-AVERAGE (FIRST COMPUTE F1 THEN TAKE AVG)==== %
TRAINPROP.F1=2.*(TRAINPROP.REC .* TRAINPROP.PREC) ./ (TRAINPROP.REC + TRAINPROP.PREC) ; % COMPUTE EACH F1 
TRAINPROP.F1(isnan(TRAINPROP.F1') & sumctest>0) = 0; % F1 VALUES THAT ARE ISNAN = 0
TRAINPROP.macroF1 = mean(TRAINPROP.F1(testc));
TRAINPROP.macroRECALL = mean(TRAINPROP.REC(testc));
TRAINPROP.macroPREC = mean(TRAINPROP.PREC(testc));

% ===== SAVE F1, PRECISION, AND RECALL, IN STATS
stats.TrainProp_microF1 = TRAINPROP.microF1;                    % MICRO-F1
stats.TrainProp_macroF1 = TRAINPROP.macroF1;                    % MACRO-F1

stats.TrainProp_microPrecision = TRAINPROP.total_precision ;    % MICRO-PRECISION
stats.TrainProp_macroPrecision = TRAINPROP.macroPREC;           % MACRO-PRECISION

stats.TrainProp_microRecall = TRAINPROP.total_recall;           % MICRO-RECALL
stats.TrainProp_macroRecall = TRAINPROP.macroRECALL;            % MACRO-RECALL

%% *AT OPTIMIZED BREAK-EVEN POINT (FOR EACH LABEL) * :  GO OVER EACH TEST-LABEL, AND COMPUTE MICRO-STATS BASED ON BREAK-EVEN POINT LOCATIONS
% COMPUTE MICRO-AVERAGE OF F1-VALUES AT EACH OF THE BREAK-EVEN POINTS
BEP.TP = zeros(nc,1);
BEP.FP = zeros(nc,1);
BEP.TN = zeros(nc,1);
BEP.FN = zeros(nc,1);

for c = testc
    
    loc=BEP.BEP_Loc(c);

    BEP.TP(c) = counts.TP(loc,c);
    BEP.FP(c) = counts.FP(loc,c);
    BEP.TN(c) = counts.TN(loc,c);
    BEP.FN(c) = counts.FN(loc,c);

end


% ===== GET THE MICRO-AVG AND MACRO-AVG OF LABEL PERFORMANCE WHEN CALIBRATED TO TRUE #   ====
BEP.sumTP = sum(BEP.TP) ;
BEP.sumFP = sum(BEP.FP) ;
BEP.sumTN = sum(BEP.TN) ;
BEP.sumFN = sum(BEP.FN) ;

BEP.FPR = BEP.FP ./ (BEP.FP + BEP.TN) ;
BEP.TPR = BEP.TP ./ (BEP.TP + BEP.FN) ; 
BEP.REC = BEP.TP ./ (BEP.TP + BEP.FN) ;
BEP.PREC = BEP.TP ./ (BEP.TP + BEP.FP) ;


% ===== GET A MICRO-AVERAGE (FIRST TOTAL THE COUNTS)==== %
BEP.total_precision = BEP.sumTP ./ (BEP.sumTP + BEP.sumFP) ;
BEP.total_recall = BEP.sumTP ./ (BEP.sumTP + BEP.sumFN) ;
% COMPUTE THE MICRO-F1 ACROSS ALL LOCATIONS
BEP.total_F1 = 2.*(BEP.total_recall .* BEP.total_precision) ./ (BEP.total_recall + BEP.total_precision) ;
BEP.microF1 = BEP.total_F1;

% ===== GET A MACRO-AVERAGE (FIRST COMPUTE F1 THEN TAKE AVG)==== %
BEP.F1=2.*(BEP.REC .* BEP.PREC) ./ (BEP.REC + BEP.PREC) ; % COMPUTE EACH F1 
BEP.F1(isnan(BEP.F1') & sumctest>0) = 0; % F1 VALUES THAT ARE ISNAN = 0
BEP.macroF1 = mean(BEP.F1(testc));
BEP.macroRECALL = mean(BEP.REC(testc));
BEP.macroPREC = mean(BEP.PREC(testc));

% ===== SAVE F1, PRECISION, AND RECALL, IN STATS
stats.BEP_microF1 = BEP.microF1;                    % MICRO-F1
stats.BEP_macroF1 = BEP.macroF1;                    % MACRO-F1

stats.BEP_microPrecision = BEP.total_precision ;    % MICRO-PRECISION
stats.BEP_macroPrecision = BEP.macroPREC;           % MACRO-PRECISION

stats.BEP_microRecall = BEP.total_recall;           % MICRO-RECALL
stats.BEP_macroRecall = BEP.macroRECALL;            % MACRO-RECALL

%% FOR EACH **TESTING** LABEL SPARSITY GET MACRO-AVG AND MICRO-AVG AUC STATISTICS

sumc = full(sumctest(1:nc));
sumc(~ismember(1:nc,testc))=0;
[uqn dummy uqnidx]=unique(sumc);

if any(~ismember(1:nc,testc))
    uqn = uqn(find(uqn));
    uqnidx = uqnidx-1;
end

numuqn = length(uqn);
fstats.microAUC_ROC = zeros(1,numuqn);
fstats.macroAUC_PR = zeros(1,numuqn);
fstats.microAUC_ROC = zeros(1,numuqn);

% ============== LOOP OVER ALL UNIQUE TRAINING-FREQUENCIES ============== %
for ii = 1 : length(uqn)
    
    ntotal_xroc = zeros(nd+2,1);
    ntotal_yroc = zeros(nd+2,1);
    ntotal_xroc(end) = 1;
    ntotal_yroc(end) = 1;

    % GET INFO FOR THIS SUMC-VAL
    n = uqn(ii);
    nidx = find(uqnidx==ii); % Get the locations of all labels with n-sparsity
    
    fstats.ntrain(ii) = n;  % Store the index of the label-frequency value for c
    
    
    %aucnidx = stats.AUC_ROC(nidx);

    % GET THE MACRO-AVG FOR THE STATISTICS OF LABELS WITH N-SPARSITY (N-VALS)
    fstats.macroAUC_ROC(ii) = mean(stats.AUC_ROC(nidx));
    fstats.macroAUC_PR(ii) = mean(stats.AUC_PR(nidx));

    
    % GET THE MICRO-AVG FOR THE N-VALS
    
    ntotalTP = sum(counts.TP(:,nidx),2);
    ntotalFP = sum(counts.FP(:,nidx),2);
    ntotalFN = sum(counts.FN(:,nidx),2);
    ntotalTN = sum(counts.TN(:,nidx),2);

    ntotal_TPR = ntotalTP ./ (ntotalTP + ntotalFN);
    ntotal_FPR = ntotalFP ./ (ntotalFP + ntotalTN);
    
    ntotal_xroc(2:1+nd) = ntotal_FPR;
    ntotal_yroc(2:1+nd) = ntotal_TPR;

    ntotal_AUC = 1-trapz(ntotal_yroc,ntotal_xroc);
    
    fstats.microAUC_ROC(ii) = ntotal_AUC;
        
    
    % ========== GET THE MACRO F1 VALUES AT CALIBRATED P LOCATIONS & BEP LOCATIONS======== %
    % MACRO AVERAGES JUST TAKE AVERAGE OF VALUES ALREADY COMPUTED
    fstats.BEP_MacroF1(ii) = mean(BEP.F1(nidx)) ;
    fstats.Calib_MacroF1(ii) = mean(CALIBP.F1(nidx)) ;
    fstats.TrainProp_MacroF1(ii) = mean(TRAINPROP.F1(nidx)) ;
    
    % ========== GET THE MICRO F1 VALUES AT CALIBRATED P LOCATIONS & BEP LOCATIONS======== %
    
    % === GET A BEP MICRO-AVERAGE (FIRST TOTAL THE COUNTS)  == %
    fstats.BEP.total_precision(ii) = sum(BEP.TP(nidx)) ./ ( sum(BEP.TP(nidx)) + sum(BEP.FP(nidx)) ) ;   % F1 ACROSS LABEL-FREQ
    fstats.BEP.total_recall(ii) = sum(BEP.TP(nidx)) ./ (sum(BEP.TP(nidx)) + sum(BEP.FN(nidx))) ;       % F1 ACROSS LABEL-FREQ
    fstats.BEP.total_F1(ii) = 2.*(fstats.BEP.total_recall(ii) .* fstats.BEP.total_precision(ii) ) ./ (fstats.BEP.total_recall(ii) + fstats.BEP.total_precision(ii)) ; % F1 ACROSS LABEL-FREQ
    if isnan(fstats.BEP.total_F1(ii)); fstats.BEP.total_F1(ii)=0; end; % DEAL WITH NAN'S
    fstats.BEP_MicroF1(ii) = fstats.BEP.total_F1(ii);
    
    % == GET A CALIBRATED (TRUE-POS = POS-PREDS) MICRO-AVERAGE (FIRST TOTAL THE COUNTS)  == %
    fstats.CALIBP.total_precision(ii) = sum(CALIBP.TP(nidx)) ./ ( sum(CALIBP.TP(nidx)) + sum(CALIBP.FP(nidx)) ) ;   % F1 ACROSS LABEL-FREQ
    fstats.CALIBP.total_recall(ii) = sum(CALIBP.TP(nidx)) ./ (sum(CALIBP.TP(nidx)) + sum(CALIBP.FN(nidx))) ;       % F1 ACROSS LABEL-FREQ
    fstats.CALIBP.total_F1(ii) = 2.*(fstats.CALIBP.total_recall(ii) .* fstats.CALIBP.total_precision(ii) ) ./ (fstats.CALIBP.total_recall(ii) + fstats.CALIBP.total_precision(ii)) ; % F1 ACROSS LABEL-FREQ
    if isnan(fstats.CALIBP.total_F1(ii)); fstats.CALIBP.total_F1(ii)=0; end; % DEAL WITH NAN'S
    fstats.Calib_MicroF1(ii) = fstats.CALIBP.total_F1(ii);
    
    % == GET A TRAININ-PROP CALIBRATED (EXPECTED POS = POS-PREDS) MICRO-AVERAGE (FIRST TOTAL THE COUNTS)  == %
    fstats.TRAINPROP.total_precision(ii) = sum(TRAINPROP.TP(nidx)) ./ ( sum(TRAINPROP.TP(nidx)) + sum(TRAINPROP.FP(nidx)) ) ;   % F1 ACROSS LABEL-FREQ
    fstats.TRAINPROP.total_recall(ii) = sum(TRAINPROP.TP(nidx)) ./ (sum(TRAINPROP.TP(nidx)) + sum(TRAINPROP.FN(nidx))) ;       % F1 ACROSS LABEL-FREQ
    fstats.TRAINPROP.total_F1(ii) = 2.*(fstats.TRAINPROP.total_recall(ii) .* fstats.TRAINPROP.total_precision(ii) ) ./ (fstats.TRAINPROP.total_recall(ii) + fstats.TRAINPROP.total_precision(ii)) ; % F1 ACROSS LABEL-FREQ
    if isnan(fstats.TRAINPROP.total_F1(ii)); fstats.TRAINPROP.total_F1(ii)=0; end; % DEAL WITH NAN'S
    fstats.TrainProp_MicroF1(ii) = fstats.TRAINPROP.total_F1(ii);
   
end


