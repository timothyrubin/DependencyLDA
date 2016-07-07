#include "mex.h"
#include "cokus.cpp"

/*
 Syntax
   [ WP , DP , Z ] = GibbsSamplerLDA( WS , DS , TAGSET , N , ALPHA , BETA , SEED , OUTPUT )

 Syntax
   [ WP , DP , Z ] = GibbsSamplerLDA( WS , DS , TAGSET , N , ALPHA , BETA , SEED , OUTPUT , ZIN )
*/

void GibbsSamplerLDA( double ALPHA, double BETA, int W, int T, int D, int NN, int OUTPUT, int n, 
                       int *z, int *d, int *w, int *wp, int *ztot, int *order, double *probs, int startcond,
                       mwIndex *irtd, mwIndex *jctd, double *srdp, mwIndex *irdp, mwIndex *jcdp )
{
  int wi,di,i,ii,j,topic, rp, temp, iter, wioffset, dioffset, rowindex, TAVAIL;
  int n1,n2,ftopic;
  double totprob, WBETA, r, max;

  if (startcond == 1) {
      /* start from previously saved state */
      for (i=0; i<n; i++)
      {
          wi = w[ i ];
          di = d[ i ];
          topic = z[ i ];
          wp[ wi*T + topic ]++; /* increment wp count matrix */

          /* increment sparse dp count matrix */
          n1 = (int) *( jcdp + di  );
          n2 = (int) *( jcdp + di + 1 );         
          TAVAIL = (n2-n1);
          ftopic = -1;
          rowindex = -1;
          while ((ftopic != topic) & (rowindex < TAVAIL-1)) {
             rowindex++;
             ftopic = (int) *( irdp + n1 + rowindex );
          }
          if (ftopic != topic) mexErrMsgTxt("Error....(10)");          
          srdp[ n1 + rowindex ]++;
          
          ztot[ topic ]++; /* increment ztot matrix */
      }
  }
  
  if (startcond == 0) {
  /* random initialization */
      if (OUTPUT==2) mexPrintf( "Starting Random initialization\n" );
      for (i=0; i<n; i++)
      {
          wi = w[ i ];
          di = d[ i ];
          
          /* what are the available tags? */
          /* look up the start and end index for tags associated with this document */
          n1 = (int) *( jctd + di     );
          n2 = (int) *( jctd + di + 1 );         
          TAVAIL = (n2-n1);
                    
          /* pick a random topic 0..TAVAIL-1 */
          rowindex = (int) ( (double) randomMT() * (double) TAVAIL / (double) (4294967296.0 + 1.0) );
          
          /* convert into a tag */
          topic = (int) *( irtd + n1 + rowindex );  
          
          z[ i ] = topic; /* assign this word token to this topic */
          wp[ wi*T + topic ]++; /* increment wp count matrix */
          ztot[ topic ]++; /* increment ztot matrix */
          
          /* increment sparse dp count matrix */
          n1 = (int) *( jcdp + di  );
          n2 = (int) *( jcdp + di + 1 );         
          TAVAIL = (n2-n1);
          ftopic = -1;
          rowindex = -1;
          while ((ftopic != topic) & (rowindex < TAVAIL-1)) {
             rowindex++;
             ftopic = (int) *( irdp + n1 + rowindex );
          }
          if (ftopic != topic) {
              mexPrintf( "n1 = %d n2=%d\n" , n1 , n2 );
              mexPrintf( "n1 = %d n2=%d  (TD)\n" , (int) *( jctd + di     ) , (int) *( jctd + di + 1 ) );
              mexPrintf( "TAVAIL=%d rowindex=%d topic=%d di=%d\n" , TAVAIL , rowindex , topic , di);
              for (rowindex=0; rowindex < TAVAIL; rowindex++)
              {
                 mexPrintf( "rowindex=%d  topic=%d topic2=%d \n" , rowindex , (int) *( irdp + n1 + rowindex ) , (int) *( irtd + n1 + rowindex )); 
              }
              mexErrMsgTxt("Error....(11)");
              
          }
          srdp[ n1 + rowindex ]++;
      }
  }
    
  if (OUTPUT==2) mexPrintf( "Determining random order update sequence\n" );
  
  for (i=0; i<n; i++) order[i]=i; /* fill with increasing series */
  for (i=0; i<(n-1); i++) {
      /* pick a random integer between i and nw */
      rp = i + (int) ((double) (n-i) * (double) randomMT() / (double) (4294967296.0 + 1.0));
      
      /* switch contents on position i and position rp */
      temp = order[rp];
      order[rp]=order[i];
      order[i]=temp;
  }
  
  /*for (i=0; i<n; i++) mexPrintf( "i=%3d order[i]=%3d\n" , i , order[ i ] ); */
  WBETA = (double) (W*BETA);
  for (iter=0; iter<NN; iter++) {
      if (OUTPUT >=1) {
          if ((iter % 10)==0) mexPrintf( "\tIteration %d of %d\n" , iter , NN );
          if ((iter % 10)==0) mexEvalString("drawnow;");
      }
      for (ii = 0; ii < n; ii++) {
          i = order[ ii ]; /* current word token to assess */
          
          wi  = w[i]; /* current word index */
          di  = d[i]; /* current document index   */
          topic = z[i]; /* current topic assignment to word token */
          ztot[topic]--;  /* substract this from counts */
          
          wioffset = wi*T;
          dioffset = di*T;
          
          wp[wioffset+topic]--;
          
          /* decrement sparse dp count matrix */
          n1 = (int) *( jcdp + di  );
          n2 = (int) *( jcdp + di + 1 );         
          TAVAIL = (n2-n1);
          ftopic = -1;
          rowindex = -1;
          while ((ftopic != topic) & (rowindex < TAVAIL-1)) {
             rowindex++;
             ftopic = (int) *( irdp + n1 + rowindex );
          }
          if (ftopic != topic) mexErrMsgTxt("Error....(12)");          
          srdp[ n1 + rowindex ]--;
          
          
          
          
          /*mexPrintf( "(1) Working on ii=%d i=%d wi=%d di=%d topic=%d wp=%d dp=%d\n" , ii , i , wi , di , topic , wp[wi+topic*W] , dp[wi+topic*D] ); */
          n1 = (int) *( jctd + di     );
          n2 = (int) *( jctd + di + 1 );         
          TAVAIL = (n2-n1);
          
          totprob = (double) 0;
          for (j = 0; j < TAVAIL; j++) 
          {
              topic = (int) *( irtd + n1 + j ); 
              probs[j] = ((double) wp[ wioffset+topic ] + (double) BETA)/
                         ( (double) ztot[topic]+ (double) WBETA) *
                         ( (double) srdp[ n1 + j ] + (double) ALPHA);
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
           
          if (j>=TAVAIL) mexErrMsgTxt("Wrong value sampled");
          
          topic = (int) *( irtd + n1 + j ); 
          
          z[i] = topic; /* assign current word token i to topic j */
          wp[wioffset + topic ]++; /* and update counts */
          ztot[topic]++;
          
          /* increment sparse dp count matrix */
          n1 = (int) *( jcdp + di  );
          n2 = (int) *( jcdp + di + 1 );         
          TAVAIL = (n2-n1);
          ftopic = -1;
          rowindex = -1;
          while ((ftopic != topic) & (rowindex < TAVAIL-1)) {
             rowindex++;
             ftopic = (int) *( irdp + n1 + rowindex );
          }
          if (ftopic != topic) mexErrMsgTxt("Error....(13)");          
          srdp[ n1 + rowindex ]++;
          
          
          /*mexPrintf( "(2) Working on ii=%d i=%d wi=%d di=%d topic=%d wp=%d dp=%d\n" , ii , i , wi , di , topic , wp[wi+topic*W] , dp[wi+topic*D] ); */
      }
  }
}

/*
 Syntax
   [ WP , DP , Z ] = GibbsSamplerLDA( WS , DS , TAGSET , N , ALPHA , BETA , SEED , OUTPUT )

 Syntax
   [ WP , DP , Z ] = GibbsSamplerLDA( WS , DS , TAGSET , N , ALPHA , BETA , SEED , OUTPUT , ZIN )
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
                 const mxArray *prhs[])
{
  double *srwp, *srdp, *srtd, *probs, *Z, *WS, *DS, *ZIN;
  double ALPHA,BETA;
  mwIndex *irwp, *jcwp, *irdp, *jcdp, *irtd, *jctd;
  int *z,*d,*w, *order, *wp, *ztot;
  int W,T,D,NN,SEED,OUTPUT, nzmax, nzmaxwp, nzmaxdp, ntokens;
  int i,j,c,n,nt,wi,di, startcond, n1,n2, TAVAIL, topic;
  
  /* Check for proper number of arguments. */
  if (nrhs < 8) {
    mexErrMsgTxt("At least 8 input arguments required");
  } else if (nlhs < 3) {
    mexErrMsgTxt("3 output arguments required");
  }
  
  startcond = 0;
  if (nrhs == 9) startcond = 1;
  
  /* process the input arguments */
  if (mxIsDouble( prhs[ 0 ] ) != 1) mexErrMsgTxt("WS input vector must be a double precision matrix");
  if (mxIsDouble( prhs[ 1 ] ) != 1) mexErrMsgTxt("DS input vector must be a double precision matrix");
  
  /* pointer to word indices */
  WS = mxGetPr( prhs[ 0 ] );
     
  /* pointer to document indices */
  DS = mxGetPr( prhs[ 1 ] );
  
  /* get the number of tokens */
  ntokens = (int) mxGetM( prhs[ 0 ] ) * (int) mxGetN( prhs[ 0 ] );
  
  
  if (ntokens == 0) mexErrMsgTxt("WS vector is empty"); 
  if (ntokens != ( mxGetM( prhs[ 1 ] ) * mxGetN( prhs[ 1 ] ))) mexErrMsgTxt("WS and DS vectors should have same number of entries");
  
  /* Input Sparse-Tag-Document Matrix */
  srtd  = mxGetPr(prhs[2]);
  irtd = mxGetIr(prhs[2]);
  jctd = mxGetJc(prhs[2]);
  nzmaxdp = (int) mxGetNzmax(prhs[2]); /* number of nonzero entries in tag-document matrix */
   
  NN    = (int) mxGetScalar(prhs[3]);
  if (NN<0) mexErrMsgTxt("Number of iterations must be positive");
  
  ALPHA = (double) mxGetScalar(prhs[4]);
  if (ALPHA<=0) mexErrMsgTxt("ALPHA must be greater than zero");
  
  BETA = (double) mxGetScalar(prhs[5]);
  if (BETA<=0) mexErrMsgTxt("BETA must be greater than zero");
  
  SEED = (int) mxGetScalar(prhs[6]);
  
  OUTPUT = (int) mxGetScalar(prhs[7]);
  
  if (startcond == 1) {
      ZIN = mxGetPr( prhs[ 8 ] );
      if (ntokens != ( mxGetM( prhs[ 8 ] ) * mxGetN( prhs[ 8 ] ))) mexErrMsgTxt("WS and ZIN vectors should have same number of entries");
  }
  
  /* seeding */
  seedMT( 1 + SEED * 2 ); /* seeding only works on uneven numbers */
    
  /* allocate memory */
  z  = (int *) mxCalloc( ntokens , sizeof( int ));
  
  if (startcond == 1) {
     for (i=0; i<ntokens; i++) z[ i ] = (int) ZIN[ i ] - 1;   
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
  
  W = 0;
  D = 0;
  for (i=0; i<n; i++) {
     if (w[ i ] > W) W = w[ i ];
     if (d[ i ] > D) D = d[ i ];
  }
  W = W + 1;
  D = D + 1;
   
  /* Number of topics is based on number of tags in sparse tag-document matrix */
  T  = (int) mxGetM( prhs[ 2 ] );
  
  /* check number of docs in sparse tag-document matrix */
  if (D != (int) mxGetN( prhs[ 2 ])) mexErrMsgTxt("Mismatch in number of documents in DS vector TAGSET sparse matrix");
  
  ztot  = (int *) mxCalloc( T , sizeof( int ));
  probs  = (double *) mxCalloc( T , sizeof( double ));
  wp  = (int *) mxCalloc( T*W , sizeof( int ));
  

  /* create sparse DP matrix that is T x D and not the other way around !!!! */
  plhs[1] = mxCreateSparse( T,D,nzmaxdp,mxREAL);
  srdp  = mxGetPr(plhs[1]);
  irdp = mxGetIr(plhs[1]);
  jcdp = mxGetJc(plhs[1]);
  
  /* now copy the structure from TD over to DP */
  for (i=0; i<D; i++) {
     n1 = (int) *( jctd + i     );
     n2 = (int) *( jctd + i + 1 );
     
     /* copy over the row-index start and end indices */
     *( jcdp + i ) = (int) n1;
    
     /* number of available topics for this document */
     TAVAIL = (n2-n1);     
     for (j = 0; j < TAVAIL; j++) {
         topic = (int) *( irtd + n1 + j );
         *( irdp + n1 + j ) = topic;
         *( srdp + n1 + j ) = 0; /* initialize DP counts with ZERO */
     }        
  }
  /* copy over final column indices */
  n1 = (int) *( jctd + D     );
  *( jcdp + D ) = (int) n1;
  

  if (OUTPUT==2) {
      mexPrintf( "Running LDA Gibbs Sampler Version 1.0\n" );
      if (startcond==1) mexPrintf( "Starting from previous state ZIN\n" );
      mexPrintf( "Arguments:\n" );
      mexPrintf( "\tNumber of words      W = %d\n"    , W );
      mexPrintf( "\tNumber of docs       D = %d\n"    , D );
      mexPrintf( "\tNumber of tags       T = %d\n"    , T );
      mexPrintf( "\tNumber of iterations N = %d\n"    , NN );
      mexPrintf( "\tHyperparameter   ALPHA = %4.4f\n" , ALPHA );
      mexPrintf( "\tHyperparameter    BETA = %4.4f\n" , BETA );
      mexPrintf( "\tSeed number            = %d\n"    , SEED );
      mexPrintf( "\tNumber of tokens       = %d\n"    , ntokens );
      mexPrintf( "\tNumber of nonzeros in tag matrix  = %d\n"    , nzmaxdp );
  }
  
  /* run the model */
  GibbsSamplerLDA( ALPHA, BETA, W, T, D, NN, OUTPUT, n, z, d, w, wp, ztot, order, probs, startcond,
                   irtd, jctd, srdp, irdp, jcdp );
  
  /* convert the full wp matrix into a sparse matrix */
  nzmaxwp = 0;
  for (i=0; i<W; i++) {
     for (j=0; j<T; j++)
         nzmaxwp += (int) ( *( wp + j + i*T )) > 0;
  }  
  
  /* MAKE THE WP SPARSE MATRIX */
  plhs[0] = mxCreateSparse( W,T,nzmaxwp,mxREAL);
  srwp  = mxGetPr(plhs[0]);
  irwp = mxGetIr(plhs[0]);
  jcwp = mxGetJc(plhs[0]);  
  n = 0;
  for (j=0; j<T; j++) {
      *( jcwp + j ) = n;
      for (i=0; i<W; i++) {
         c = (int) *( wp + i*T + j );
         if (c >0) {
             *( srwp + n ) = c;
             *( irwp + n ) = i;
             n++;
         }
      }    
  }  
  *( jcwp + T ) = n;    
   
  plhs[ 2 ] = mxCreateDoubleMatrix( 1,ntokens , mxREAL );
  Z = mxGetPr( plhs[ 2 ] );
  for (i=0; i<ntokens; i++) Z[ i ] = (double) z[ i ] + 1;
}
