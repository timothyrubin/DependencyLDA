#include "mex.h"
#include "cokus.cpp"


void GibbsSamplerLDA2( int W, int T, int D, int NN, int n, int *z, int *d, int *w, int *order, int *dp, double *probs, double *pcw, double *cdalpha, int startcond, int OUTPUT){

  int wi,di,i,ii,j,topic, rp, temp, iter, wioffset, dioffset, rowindex, TAVAIL,k, sum;
  int n1,n2,ftopic;
  double totprob, r, max;
  
  if (startcond == 1) {
    /* start from previously saved state */
    for (i=0; i<n; i++){
      wi = w[ i ];
      di = d[ i ];
      topic = z[ i ];
      dp[ di*T + topic ] ++; 
    }
  }
  
  if (startcond == 0) {
    /* random initialization */
    if (OUTPUT==2) mexPrintf( "Starting Random initialization\n" );
    for (i=0; i<n; i++) {
      wi = w[ i ];
      di = d[ i ];
      
      /* pick a random topic 0..T */
      topic = (int) ( (double) randomMT() * (double) T / (double) (4294967296.0 + 1.0) );
      z[ i ] = topic;
      dp[ di*T + topic ]++;
    }
  }
  
  if (OUTPUT==2) mexPrintf( "Determining random order update sequence\n" );
  
  for (i=0; i<n; i++) order[i]=i; /* fill with increasing series  */
  for (i=0; i<(n-1); i++) {
    /* pick a random integer between i and nw */
    rp = i + (int) ((double) (n-i) * (double) randomMT() / (double) (4294967296.0 + 1.0));
    
    /* switch contents on position i and position rp */
    temp = order[rp];
    order[rp]=order[i];
    order[i]=temp;
  }
  
  for (iter=0; iter<NN; iter++) {
    if (OUTPUT >=1) {
      if ((iter % 10)==0) mexPrintf( "\tIteration %d of %d\n" , iter , NN );
      if ((iter % 10)==0) mexEvalString("drawnow;");
    }
    for (ii = 0; ii < n; ii++) {
      i = order[ ii ]; /* current word token to assess */
      
      wi  = w[i]; /* current word index */
      di  = d[i]; /* current document index  */
      topic = z[i]; /* current topic assignment to word token */
      
      wioffset = wi*T;
      dioffset = di*T;
      
      /* decrement sparse dp count matrix */
      dp[ dioffset + topic ]--;
      
      totprob = (double) 0;
      for (j = 0; j < T; j++) {
		probs[j] = (double) pcw[wi*T + j] * ((double) dp[ dioffset + j ] + (double) cdalpha[ di*T + j] );
		totprob += probs[j];
      }
      
      /* sample a topic from the distribution */
      r = (double) totprob * (double) randomMT() / (double) 4294967296.0;
      max = probs[0];
      j = 0;
      while (r>max) {
		j++;
		max += probs[j];
      }
      
      if (j>=T) mexErrMsgTxt("Wrong value sampled");
      topic = j;
      
      z[i] = topic; /*  assign current word token i to topic j */
      dp[ dioffset + topic ]++; /* and update counts */
    }
  }
}


/* Syntax
 [ PD, ZOUT ] = GibbsSamplerLDA2( WS, DS, DCAlpha, PWC, NITER, SEED, OUTPUT )
 [ PD, ZOUT ] = GibbsSamplerLDA2( WS, DS, DCAlpha, PWC, NITER, SEED, OUTPUT, ZIN)

 Note: Although I store the output as a DxT matrix, when you save it to a Matlab matrix
 they do column first so its returned as a TxD matrix  */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[])
{
  double *probs, *cdalpha, *pcw; 
  double *Z, *WS, *DS, *ZIN, *CDAlpha, *PCW, *DP, *ZOUT;
  int *z,*d,*w, *order, *dp;
  int W,T,D,NN,SEED,OUTPUT, ntokens;
  int i,j,c,n,nt,wi,di, startcond, n1,n2, TAVAIL, topic;
  int no_topics, no_docs, no_words;
  
  /* Check for proper number of arguments. */
  if (nrhs < 7) {
    mexErrMsgTxt("At least 7 input arguments required");
  } else if (nlhs != 2) {
    mexErrMsgTxt("2 output arguments required");
  }
  
  startcond = 0;
  if (nrhs == 8) startcond = 1;
  
  /* process the input arguments */
  if (mxIsDouble( prhs[ 0 ] ) != 1) mexErrMsgTxt("WS input vector must be a double precision matrix");
  if (mxIsDouble( prhs[ 1 ] ) != 1) mexErrMsgTxt("DS input vector must be a double precision matrix");
  if (mxIsDouble( prhs[ 2 ] ) != 1) mexErrMsgTxt("CDAlpha must be a double precision matrix");
  if( mxIsDouble( prhs[ 3 ] ) != 1) mexErrMsgTxt("PWD must be a double precision matrix");
  
  WS      = mxGetPr( prhs[ 0 ] );  /* pointer to word indices */
  DS      = mxGetPr( prhs[ 1 ] );  /* pointer to document indices  */
  CDAlpha = mxGetPr( prhs[ 2 ] );  /* pointer to Dirichlet hyerparameters  */
  PCW     = mxGetPr( prhs[ 3 ] );  /* pointer to learned topics  */

  /* get the number of tokens   */
  ntokens = (int) mxGetM( prhs[ 0 ] ) * (int) mxGetN( prhs[ 0 ] );  
  if (ntokens == 0) mexErrMsgTxt("WS vector is empty"); 
  if (ntokens != ( mxGetM( prhs[ 1 ] ) * mxGetN( prhs[ 1 ] ))) mexErrMsgTxt("WS and DS vectors should have same number of entries");
  
  /* get the number of topics, docs and words  */
  no_docs = (int) mxGetM( prhs[ 2 ]);
  no_topics = (int) mxGetN( prhs[ 2 ] );
  no_words = (int) mxGetM( prhs[ 3 ] );
  if (no_topics != mxGetN( prhs[ 3 ])) mexErrMsgTxt("DCalpha and PWC have differing C values");

  /* get number of iterations  */
  NN  = (int) mxGetScalar(prhs[4]);
  if (NN<0) mexErrMsgTxt("Number of iterations must be positive");
  
  /* get seed   */
  SEED = (int) mxGetScalar(prhs[5]);
  OUTPUT = (int) mxGetScalar(prhs[6]);
  
  if (startcond == 1) {
    ZIN = mxGetPr( prhs[ 7 ] );
    if (ntokens != ( mxGetM( prhs[ 7 ] ) * mxGetN( prhs[ 7 ] ))) mexErrMsgTxt("WS and ZIN vectors should have same number of entries");
  }
  
  /* seeding   */
  seedMT( 1 + SEED * 2 ); /* seeding only works on uneven numbers */
  
  /* allocate memory */
  z  = (int *) mxCalloc( ntokens , sizeof( int ));

  if (startcond == 1) {
    for (i=0; i<ntokens; i++) z[ i ] = (int) ZIN[ i ] - 1;   
  }

  /* copy over the PWC matrix into internal format */
  /*/ This is passed in as a WxC matrix but stupid matlab stores it column first */
  /* I'm going to store it row first */
  pcw = (double *) mxCalloc( no_words*no_topics, sizeof( double ));
  for(i=0; i < no_words; i++){
	for(j=0; j < no_topics; j++){
		pcw[i*no_topics + j] = (double)PCW[j*no_words + i];
	}
  }

  /* copy over the DCAlpha matrix into internal format */
  /* I'm going to store it row first */
  cdalpha = (double *) mxCalloc( no_docs*no_topics, sizeof( double ));
  for( i = 0; i < no_docs; i++){
	for( j = 0; j < no_topics; j++){
		cdalpha[ i*no_topics + j] = (double)CDAlpha[ j*no_docs + i];
	}
  }

  d  = (int *) mxCalloc( ntokens , sizeof( int ));
  w  = (int *) mxCalloc( ntokens , sizeof( int ));
  order  = (int *) mxCalloc( ntokens , sizeof( int ));  
  
  /* copy over the word and document indices into internal format */
  for (i=0; i<ntokens; i++) {
     w[ i ] = (int) WS[ i ] - 1;
     d[ i ] = (int) DS[ i ] - 1;
  }

  n = ntokens;  
  W = no_words;
  D = no_docs;
  T  = no_topics;
  probs  = (double *) mxCalloc( T , sizeof( double ));
  dp    = (int *) mxCalloc( no_docs*no_topics, sizeof(int));

  if (OUTPUT==2) {
      mexPrintf( "Running LDA Gibbs Sampler Version 1.0\n" );
      if (startcond==1) mexPrintf( "Starting from previous state ZIN\n" );
      mexPrintf( "Arguments:\n" );
      mexPrintf( "\tNumber of words      W = %d\n"    , W );
      mexPrintf( "\tNumber of docs       D = %d\n"    , D );
      mexPrintf( "\tNumber of tags       T = %d\n"    , T );
      mexPrintf( "\tNumber of iterations N = %d\n"    , NN );
      mexPrintf( "\tSeed number            = %d\n"    , SEED );
      mexPrintf( "\tNumber of tokens       = %d\n"    , ntokens );
  }
  /* run the model */
  GibbsSamplerLDA2( W, T, D, NN, n, z, d, w, order, dp, probs, pcw, cdalpha, startcond, OUTPUT);
  

  plhs[ 0 ] = mxCreateDoubleMatrix( T, D, mxREAL );
  DP = mxGetPr( plhs[ 0 ] );
  for (i=0; i < no_docs*no_topics; i++){
    DP[ i ] = (double) dp[ i ] ;
  }

  plhs[ 1 ] = mxCreateDoubleMatrix(1, ntokens, mxREAL);
  ZOUT = mxGetPr( plhs[ 1 ] );
  for(i=0; i < ntokens; i++){
	ZOUT[i] = (double)z[i]+1;
  }




}
