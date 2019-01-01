#include "mex.h" 
#include <stdlib.h> 
#include <time.h>

void get_N(double UL[], double UFG[], double myMap[], double rand_nums[],
        int user_num, int loc_num, int myu, int mya, double maginal, 
        double* N, double* myb_res) 
{
    int i, myb;
    double S_ua, S_a_gInf, S_ub, S_b_gInf;
    double tmp;
    // srand((unsigned)time(NULL));
    
    S_ua = UL[user_num * mya + myu];
    S_a_gInf = UFG[user_num * mya + myu];
    for (i = 0; i < loc_num; ++i)
    {
        // myb = rand() % loc_num;
        myb = (int)(rand_nums[i]);
        if(myMap[user_num * mya + myu] > myMap[user_num * myb + myu])
        {
            S_ub = UL[user_num * myb + myu];
            S_b_gInf = UFG[user_num * myb + myu];
            tmp = S_ua + S_a_gInf - S_ub - S_b_gInf;
            if (tmp < maginal)
            {
                N[0] = i + 1;
                myb_res[0] = myb + 1;
                return;
            }
        }
    }
    N[0] = loc_num;
    myb_res[0] = -1;
    return;
}

//parameters: UL, UFG, myMap, user_num, loc_num, myu, mya, marginal
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{ 
    double *N, *myb_res;
    double *UL, *UFG, *myMap, *rand_nums;
    int user_num, loc_num;
    int myu, mya;
    double maginal;
    
    if(nrhs!=9) { 
    mexErrMsgTxt("One input required."); 
    } else if(nlhs>2) { 
    mexErrMsgTxt("Too many output arguments"); 
    }
    UL = mxGetPr(prhs[0]); 
    UFG = mxGetPr(prhs[1]); 
    myMap = mxGetPr(prhs[2]); 
    rand_nums = mxGetPr(prhs[3]);
    user_num = (int)(*mxGetPr(prhs[4]));
    loc_num = (int)(*mxGetPr(prhs[5]));
    myu = (int)(*mxGetPr(prhs[6]));
    mya = (int)(*mxGetPr(prhs[7]));
    maginal = *mxGetPr(prhs[8]);
    
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    N = mxGetPr(plhs[0]);
    myb_res = mxGetPr(plhs[1]);
    get_N(UL, UFG, myMap, rand_nums, user_num, 
            loc_num, myu, mya, maginal, N, myb_res);
}