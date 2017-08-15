#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void sm_ssa(int steps, double tmax, int nstates, int initialstate, double *flat_rates, int *states, double *dwells, int *last);
void render_trace(int steps, int timesteps, int nstates, double *x, double *y, int *states, double *times, double *dwells, double *emissions);

void sm_ssa(int steps, double tmax, int nstates, int initialstate, double *flat_rates, int *states, double *dwells, int *last){

	int i,j;
	double r1,r2;
	double rates[nstates][nstates], cumulatives[nstates][nstates];
	double outrates[nstates];
	int currentstate = initialstate;
	double timetotal = 0.;

	// setup cutoffs and rates
	for(i=0;i<nstates;i++){
		outrates[i] = 0.;
		cumulatives[i][0] = 0.;
		for(j=0;j<nstates;j++){
			rates[i][j] = flat_rates[nstates*i + j];
			cumulatives[i][j] = flat_rates[nstates*i + j];
			if (j > 0){
				cumulatives[i][j] += cumulatives[i][j-1];
			}
		}
		outrates[i] = cumulatives[i][nstates-1];
	}

	// seed random num generator
	// srand(time(NULL);

	for(i=0;i<steps;i++){
		states[i] = currentstate;

		r1 = ((double)rand()) / ((double)RAND_MAX);
		r2 = ((double)rand()) / ((double)RAND_MAX);
		dwells[i] = 1./outrates[states[i]]* log(1./r1);

		if (i == 0){ // Start trace at random time.
			dwells[i]*=(((double)rand()) / ((double)RAND_MAX));
		}
		timetotal += dwells[i];

		for(j=0;j<nstates;j++){
			currentstate = j;
			if (cumulatives[states[i]][j] > r2*outrates[states[i]]) {
				break;
			}
		}
		if (timetotal > tmax) {
			last[0] = i+1;
			break;
		}
	}
}

void render_trace(int steps, int timesteps, int nstates, double *x, double *y, int *states, double *times, double *dwells, double *emissions) {
	// printf("%d %d %d %f %f %d %f %f %f\n",steps,timesteps,nstates,x[0],y[0],states[0],times[0],dwells[0],emissions[0]);

	int i,j;
	double t0,t1;
	double tau = x[1] - x[0];
	int a=0, b=0,aflag = 1,bflag=1;

	for (i=0;i<steps;i++) {
		t1 = x[i];
		t0 = t1 - tau;
		aflag = 1;
		bflag=1;
		for (j=a;j<timesteps+1;j++){
			if (times[j] > t0 && aflag) {
				a = j-1;
				aflag = 0;
			}
			if (times[j] > t1 && bflag) {
				b = j;
				bflag = 0;
			}
			if (!aflag && !bflag){
				y[i] = 0.;
				if (b-a == 1){
					y[i] += (t1-t0)/tau * emissions[states[a]];
				}
				else if (b-a > 1) {
					y[i] += (times[a+1]-t0)/tau * emissions[states[a]];
					for (j=a+1;j<b-1;j++) {
						y[i] += dwells[j]/tau * emissions[states[j]];
					}
					y[i]+= (t1 - times[b-1])/tau * emissions[states[b-1]];
				}
				break;
			}
		}
	}
}

// gcc -shared -o ./sm_ssa-linux.so -fPIC -O3 ./sm_ssa.c
// gcc -shared -o ./sm_ssa-mac.so -fPIC -O3 ./sm_ssa.c
