package twoPhaseSensitivity;


import java.util.ArrayList;
import java.util.Random;

import gurobi.*;

public class CallbackSurvey extends GRBCallback {

	algHandler alg;
	private ArrayList<GRBVar>[] y;
	private ArrayList<GRBVar>[] h;

	private ArrayList<Double>[] yhat;
	private ArrayList<Double>[] hhat;


	public CallbackSurvey(algHandler a, ArrayList<GRBVar>[] yc, ArrayList<GRBVar>[] hc) {
		alg = a;
		y = yc;
		h = hc;
		yhat = new ArrayList[alg.numNN];
		hhat = new ArrayList[alg.numNN];

		for (int i = 0; i < alg.numNN; i++) {
			yhat[i] = new ArrayList<Double>();
			hhat[i] = new ArrayList<Double>();
		}
	}

	protected void callback() {
		try {

			if(where == GRB.CB_MIPNODE) {
				if(alg.totalCuts< 1000) {
					// Capture all the variable values
					for (int i = 0; i < alg.numNN; i++) {
						for (int j = 0; j < y[i].size(); j++) {
							yhat[i].add(getNodeRel(y[i].get(j)));
						}					
					}
					for (int i = 0; i < alg.numNN; i++) {
						for (int j = 0; j < h[i].size(); j++) {
							hhat[i].add(getNodeRel(h[i].get(j)));
						}					
					}

					// Check violations
					for (int i = 0; i < alg.numNN; i++) {
						for (int l = 1; l < alg.NNs[i].numLayers-1 ; l++) {
							for (int v = 0; v < alg.NNs[i].layers[l].size(); v++) {
								Node n = alg.NNs[i].layers[l].get(v);
								// Only check violation if the binary var is not fixed and y is more than 0!
								if(yhat[i].get(n.ID)>=alg.tol && n.fixActivation==9 && yhat[i].get(n.ID)-hhat[i].get(n.ID)>0.1){
									n.numCritical++;
									n.violations+=(yhat[i].get(n.ID)-hhat[i].get(n.ID));
								}  
							}
						}
					}
					alg.totalCuts++;
				}
				else{
					abort();
				}
			}

		} catch (GRBException e) {
			//System.out.println("Error code: " + e.getErrorCode() + ". " +e.getMessage());
			//e.printStackTrace();
		}
	}



}

