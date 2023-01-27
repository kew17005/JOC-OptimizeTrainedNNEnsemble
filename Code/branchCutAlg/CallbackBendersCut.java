package branchCutAlg;


import java.util.ArrayList;
import java.util.Random;

import gurobi.*;

public class CallbackBendersCut extends GRBCallback {

	algHandler alg;
	private ArrayList<GRBVar>[] y;
	private ArrayList<GRBVar>[] z;
	
	private ArrayList<Double>[] yhat;
	private ArrayList<Double>[] zhat;


	public CallbackBendersCut(algHandler a, ArrayList<GRBVar>[] yc, ArrayList<GRBVar>[] zc) {
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

			if(where == GRB.CB_MIPSOL) {
				if(alg.totalCuts< 10000000) {
					// Capture all the variable values
					for (int i = 0; i < alg.numNN; i++) {
						alg.zval[i].clear();
						for (int j = 0; j < z[i].size(); j++) {
							if(getSolution(z[i].get(j))< 0.1) {
								alg.zval[i].add(0);
							}
							else {
								alg.zval[i].add(1);
							}
							//System.out.print(getSolution(z[i].get(j))+" ");
						}
						//System.out.println();					
					}
					System.out.println("ABOUT TO SOLVE FOR SOLUTION WITH VAL "+getDoubleInfo(GRB.CB_MIPSOL_OBJ));
					// Solve subproblem
					bendersCut cut = alg.solveSubproblem();
					if(cut != null){
					GRBLinExpr exprC = new GRBLinExpr();
					double rhs = cut.constant;
					//Objective
					for (int i = 0; i < alg.numNN; i++) {
						exprC.addTerm(-(1/(alg.numNN+0.0)), y[i].get(alg.NNs[i].numNodes-1));
					}
					for (int i = 0; i < alg.numNN; i++) {
						int cindex = 0;
						for (int l = 1; l < alg.NNs[i].numLayers-1; l++) {
							for (int v = 0; v < alg.NNs[i].layers[l].size(); v++) {
								int vindex = alg.NNs[i].layers[l].get(v).ID;
								exprC.addTerm(cut.coef0[i].get(cindex), z[i].get(vindex));
								exprC.addTerm(-cut.coef1[i].get(cindex), z[i].get(vindex));
								rhs += cut.coef1[i].get(cindex);
								cindex++;
							}
						}
					}
					if(alg.isMax==false) {addLazy(exprC, GRB.LESS_EQUAL, -rhs);}
					else {addLazy(exprC, GRB.GREATER_EQUAL, -rhs);}
					alg.totalCuts++;
				}
				else {
					// Add no good cut
					int rhs = 0;
					GRBLinExpr expr = new GRBLinExpr();
					for (int i = 0; i < 1; i++) {
						for (int j = 0; j < z[i].size(); j++) {
							if(alg.zval[i].get(j) < 0.1) {
								expr.addTerm(1, z[i].get(j));
							}
							else {
								expr.addTerm(-1, z[i].get(j));
								rhs++;
							}
						}
					}
					addLazy(expr, GRB.GREATER_EQUAL, 1-rhs);
					alg.totalCuts++;
				}	
				System.out.println("CURRENT LB = "+alg.LB+" UB = "+getDoubleInfo(GRB.CB_MIPSOL_OBJBND));	
					
				}
			}
			
		} catch (GRBException e) {
			//System.out.println("Error code: " + e.getErrorCode() + ". " +e.getMessage());
			//e.printStackTrace();
		}
		

		
		
	}

		
	



}

