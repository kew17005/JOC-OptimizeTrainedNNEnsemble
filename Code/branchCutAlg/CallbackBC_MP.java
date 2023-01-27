package branchCutAlg;


import java.util.ArrayList;
import java.util.Random;

import gurobi.*;

public class CallbackBC_MP extends GRBCallback {

	algHandler alg;
	private ArrayList<GRBVar>[] y;
	private ArrayList<GRBVar>[] z;

	private ArrayList<Double>[] yhat;
	private ArrayList<Double>[] zhat;


	public CallbackBC_MP(algHandler a, ArrayList<GRBVar>[] yc, ArrayList<GRBVar>[] zc) {
		alg = a;
		y = yc;
		z = zc;
		yhat = new ArrayList[alg.numNN];
		zhat = new ArrayList[alg.numNN];

		for (int i = 0; i < alg.numNN; i++) {
			yhat[i] = new ArrayList<Double>();
			zhat[i] = new ArrayList<Double>();
		}
	}

	protected void callback() {
		try {

			if(where == GRB.CB_MIPNODE) {
				// Get root node relaxation
				if(alg.rootNodeDone == false) {
					alg.rootBound = getDoubleInfo(GRB.CB_MIPNODE_OBJBND);
					alg.rootNodeDone = true;
					//System.out.println("WEPAAAAAAAAAAA: "+alg.rootBound);
					
				}
					
				if(alg.totalCuts< 25000) {
					// Capture all the variable values
					for (int i = 0; i < alg.numNN; i++) {
						for (int j = 0; j < y[i].size(); j++) {
							yhat[i].add(getNodeRel(y[i].get(j)));
						}					
					}
					for (int i = 0; i < alg.numNN; i++) {
						for (int j = 0; j < z[i].size(); j++) {
							zhat[i].add(getNodeRel(z[i].get(j)));
						}					
					}

					// Check if there are cuts to add for every node in every NN
					alg.numCuts = 0;
					for (int i = 0; i < alg.numNN; i++) {
						for (int l = 1; l < alg.NNs[i].numLayers-1 ; l++) {
							for (int v = 0; v < alg.NNs[i].layers[l].size(); v++) {
								Node n = alg.NNs[i].layers[l].get(v);
								// Only check violation if the binary var is not fixed!
								if(yhat[i].get(n.ID)>=alg.tol && zhat[i].get(n.ID)>=alg.tol && zhat[i].get(n.ID)<=1-alg.tol && n.fixActivation==9) {
									checkCutViolation(n,i,l);
								}  
							}
						}
					}
					
				}
				// Activate for sensitivity analysis about bounds
				//if(alg.totalCuts>=25000) {
					//System.out.println("DONE ADDING CUTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
					//alg.afterCutsBound = getDoubleInfo(GRB.CB_MIPNODE_OBJBND);
					//abort();
				//}
				
			}

		} catch (GRBException e) {
			//System.out.println("Error code: " + e.getErrorCode() + ". " +e.getMessage());
			//e.printStackTrace();
		}
	}

	private void checkCutViolation(Node n, int i, int l) throws GRBException {
		ArrayList<Integer> I = new ArrayList<>();
		ArrayList<Integer> Icomp = new ArrayList<>();
		double yval = yhat[i].get(n.ID);  // Get val for y
		double zval = zhat[i].get(n.ID);  // Get val for z
		double xval[] = new double[n.dim];
		for (int u = 0; u < n.dim; u++) {
			int index = alg.NNs[i].layers[l-1].get(u).ID;
			xval[u] = yhat[i].get(index);
		}
		// Build set I hat and its complement
		for (int u = 0; u < n.dim; u++) {
			if(n.weights[u]*xval[u] < n.weights[u]*( n.L[u]*(1-zval)  +  n.U[u]*zval ) ){
				I.add(u);
			}
			else {
				Icomp.add(u);
			}
		}
		//System.out.println("Node "+n.ID+"Set I: "+I);
		//System.out.println("Node "+n.ID+"Set Icomp: "+Icomp);

		// Check condition to see if cut is violated (paper pg28)
		double lhs = 0;
		lhs += n.bias*zval;
		for (int k = 0; k < I.size(); k++) {
			int u = I.get(k);
			lhs+= n.weights[u]*(xval[u] - n.L[u]*(1-zval) );
		}
		for (int k = 0; k < Icomp.size(); k++) {
			int u = Icomp.get(k);
			lhs+= n.weights[u]*n.U[u]*zval;
		}
		if( yval > lhs + 0.1  ) {
			//	System.out.println("Node "+n.ID+" CUT VIOLATED!!!!!");
			//	System.out.println("      yval: "+yval+"  LHS: "+lhs);

			double rhs = 0;
			for (int k = 0; k < I.size(); k++) {
				int u = I.get(k);
				rhs+= n.weights[u]*n.L[u];
			}
			GRBLinExpr expr = new GRBLinExpr();
			for (int k = 0; k < I.size(); k++) {
				int u = I.get(k);
				int index = alg.NNs[i].layers[l-1].get(u).ID;
				expr.addTerm(n.weights[u], y[i].get(index));
			}
			for (int k = 0; k < I.size(); k++) {
				int u = I.get(k);
				expr.addTerm(n.weights[u]*n.L[u], z[i].get(n.ID));
			}
			expr.addTerm(n.bias, z[i].get(n.ID));
			for (int k = 0; k < Icomp.size(); k++) {
				int u = Icomp.get(k);
				expr.addTerm(n.weights[u]*n.U[u], z[i].get(n.ID));
			}
			expr.addTerm(-1, y[i].get(n.ID));

			addCut(expr, GRB.GREATER_EQUAL, rhs);   
			alg.numCuts++;
			alg.totalCuts++;
			//System.out.println("CUT ADDED");

		}




	}



}

