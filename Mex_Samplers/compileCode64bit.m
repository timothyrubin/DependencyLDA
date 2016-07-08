% To compile:

mex GibbsSampler_TagLDA_DCAlpha_V04.c -largeArrayDims

%% This one gives error on some machines but uses a work-around in code to handle errors
mex GibbsSamplerLDA_v2.c -largeArrayDims

%%
mex GibbsSamplerLDATAGS3.c -largeArrayDims
