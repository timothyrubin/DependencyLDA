function WriteTopicsToCSV(outfilename,mat,rowstr,colstr,nrows2print)
% Input:
%   Output Filename
%   Matrix of total counts in word-x-topic matrix (typically WxT count matrix + priors)
%   Row-Strings (Word-vocab typically)
%   Col-Strings (Label-Vocab Typically)
%   N-Rows to print (Number of Rows to print for each column; defaults to min[nrows, 100])
%
% WRITES OUT A CSV FILE CONTAINING:
%   - A column label
%   - The row-strings and values, sorted by each column

%% 
if nargin < 4
    % If no column-header strings, generate a generic cell-array of headers
    colstr = cell(1,size(mat,2));
    for i = 1 : size(mat,2)
        colstr{i} = sprintf('Topic_%02d',i);
    end
end

if nargin < 5
    % If no #rows input, default to 50 or max # rows
    nrows2print = min(50, size(mat,1));
end

%% Since we are outputting to a csv file, need to remove any commas from the row or column strings
% Remove any commas from rows
for i = 1 : size(mat,1)
    rowstr{i} = strrep(rowstr{i},',','');    
end
% Remove any commas from headers
for i = 1 : size(mat,2)
    colstr{i} = strrep(colstr{i},',','');    
end
%% Get nrows and ncols and do checks
[nrows ncols] = size(mat);
assert(nrows==length(rowstr));
assert(ncols==length(colstr));

%% We have header strings, but also need header vals
% Header vals default to sum(cols) / sum(sum(cols))
headervals = sum(mat)./sum(sum(mat));

%% NORMALIZE MATRIX
mat = mat./repmat(sum(mat,1),[nrows,1]);

%% Get a matrix with the sorted values and ids
[srtvals srtids] = sort(mat,1,'descend');

%% Write the output file
fid = fopen(outfilename,'w');

% Write out Headers first
for i=1:ncols
    fprintf(fid,'%s,', colstr{i});          % Header label then comma
    fprintf(fid,'%.4f,', headervals(i));    % Header vals then comma
    %fprintf(fid,',');                      % Another comma if we want a column of spacing between topics
end
fprintf(fid,'\n');

% Go through the column and write out the sorted row-strings + values for each
for i=1:nrows2print
    for j=1:ncols
        fprintf(fid,'%s,', rowstr{srtids(i,j)});% Header label then comma
        fprintf(fid,'%0.4f,', srtvals(i,j));    % Header vals then comma
        %fprintf(fid,',');                      % Another comma if we want a column of spacing between topics
    end
    fprintf(fid,'\n'); 
end

%%
fclose(fid);

