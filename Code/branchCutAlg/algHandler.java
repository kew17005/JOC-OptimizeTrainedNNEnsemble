package branchCutAlg;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import gurobi.*;

public class algHandler {

	// Number of trees
	int numNN;
	boolean isMax;

	// Problem Data
	NN[] NNs;						//All the NNs 										 

	double primalBound;				// Obj of a feasible solution

	GRBEnv env;						// Gurobi Environment
	GRBModel model;					// MIPmodel
	double opt;						// Optimal solution
	double gap;						// MIP Gap
	double tol = 0.0001;			// Numerical tolerance
	int numCuts;
	int totalCuts;

	double startTime;				//To capture run times
	ArrayList<double[]> solutions; 	// Stores solutions for each NN 

	double maxDom;					// Maximum domain for any h 

	// For Benders
	ArrayList<Integer>[] zval;  		//Current value for the z-variables
	GRBEnv envS;						// Gurobi Environment for the sub problems
	GRBEnv envB;						// Gurobi Environment for the sub problems
	GRBModel modelS;					// LP models 
	GRBModel modelB;					// LP models 
	double UB;
	double LB;
	ArrayList<bendersCut> cutPool;		// all the cuts generated so far

	// For stats
	double lpbounds;					// Average width of the bounds via LPs
	double ipbounds;					// Average width of the bounds via IPs
	double numCritical;					// Number of critical neurons

	boolean rootNodeDone;
	double rootBound;
	double afterCutsBound;
	
	
	double timeBounding;
	double timeIP;
	// For spatial
	double delta = 0.1;
	double[] xval;

	public algHandler() throws GRBException {
		env = new GRBEnv(null);
		envS = new GRBEnv(null);
		envB = new GRBEnv(null);
		isMax = false;
		maxDom = 0;
		UB= 999999;
		LB = -999999;
		cutPool = new ArrayList<bendersCut>();
		timeBounding = 0;
		timeIP = 0;
		rootBound = 0;
		afterCutsBound = 0;
		rootNodeDone = false;
	}

	public void readInstance(String name) throws NumberFormatException, IOException {

		////////////////////////////////////////////////////// Read the data file
		File file = new File(name);
		BufferedReader bufRdr = new BufferedReader(new FileReader(file));

		// Read numNN
		String line = bufRdr.readLine();
		String[] tokens = line.split(" ");
		numNN = Integer.parseInt(tokens[3]);
		NNs = new NN[numNN];
		System.out.println("NUM NN = "+numNN);
		// Read configurations
		for (int i = 0; i < numNN; i++) {
			line = bufRdr.readLine();
			tokens = line.split(" ");
			int numLayers = tokens.length-3;
			NNs[i] = new NN(numLayers);
			for (int j = 3; j < tokens.length; j++) {
				NNs[i].config[j-3] = Integer.parseInt(tokens[j]); 	
			}
		}
		// Read each NN
		line = bufRdr.readLine();
		for (int i = 0; i < numNN; i++) {
			int nodeCount = 0;
			// Input layer
			for (int k = 0; k < NNs[i].config[0]; k++) {
				NNs[i].layers[0].add(new Node(0, nodeCount, 0));
				nodeCount++;
			}
			NNs[i].numInputs = NNs[i].config[0];
			for (int j = 1; j < NNs[i].numLayers; j++) {
				for (int k = 0; k < NNs[i].config[j]; k++) {
					line = bufRdr.readLine();
					tokens = line.split(" ");
					int numW = tokens.length-3;
					Node n = new Node(numW, nodeCount, j);
					nodeCount++;
					for (int l = 2; l < tokens.length-1; l++) {
						n.weights[l-2] = round(Double.parseDouble(tokens[l]));
					}
					n.bias = round(Double.parseDouble(tokens[tokens.length-1]));
					NNs[i].layers[j].add(n);
				}
			}
			NNs[i].numNodes = nodeCount;
			//NNs[i].print();
		}

		zval = new ArrayList[numNN];
		for (int i = 0; i < numNN; i++) {
			zval[i] = new ArrayList<Integer>();
		}

	}


	private double round(double value) {
		double rounded;
		rounded = Math.round(value*100000)/100000.0;
		return rounded;
	}

	private double roundS(double value) {
		double rounded;
		rounded = Math.round(value*100)/100.0;
		return rounded;
	}

	public void IPBigM() throws GRBException {
		env.set(GRB.IntParam.OutputFlag, 1);
		env.set(GRB.IntParam.LazyConstraints, 1);
		model = new GRBModel(env);
		double timeUsed = (System.currentTimeMillis()-startTime)/1000;
		model.set(GRB.DoubleParam.TimeLimit, 3600.0-timeUsed);
		if(isMax) {model.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(model.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(model.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, ""));}
					h[i].add(model.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					if(NNs[i].layers[l].get(v).fixActivation==9) {z[i].add(model.addVar(0, 1, 0, GRB.BINARY, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {z[i].add(model.addVar(0, 0, 0, GRB.CONTINUOUS, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {z[i].add(model.addVar(1, 1, 0, GRB.CONTINUOUS, ""));}
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = model.addVar(-1, 1, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		model.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// alpha_v=G_v = F_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				model.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				model.addConstr(expr2, GRB.EQUAL, 0, "");
			}	
			// G_t = F_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			model.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					model.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					if(NNs[i].layers[l].get(v).fixActivation==9) { // Binary variable not fixed
						double lb = -NNs[i].layers[l].get(v).lb;
						double ub = NNs[i].layers[l].get(v).ub;
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-ub, z[i].get(vindex));
						model.addConstr(expr, GRB.LESS_EQUAL, 0, "");

						GRBLinExpr expr6 = new GRBLinExpr();
						expr6.addTerm(1, y[i].get(vindex));
						model.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr2 = new GRBLinExpr();
						expr2.addTerm(1, y[i].get(vindex));
						expr2.addTerm(-1, h[i].get(vindex));
						model.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr4 = new GRBLinExpr();
						expr4.addTerm(1, y[i].get(vindex));
						expr4.addTerm(-1, h[i].get(vindex));
						expr4.addTerm(lb, z[i].get(vindex));
						model.addConstr(expr4, GRB.LESS_EQUAL, lb, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {//Node always inactive 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						model.addConstr(expr, GRB.EQUAL, 0, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {//Node always active 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-1, h[i].get(vindex));
						model.addConstr(expr, GRB.EQUAL, 0, "");
					}
				}
			}

		}

		// Solve
		model.update();
		model.setCallback(new CallbackBendersCut(this , y, z));
		model.optimize();
		if(model.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			System.out.println("		MIP is infeasible!");		
		}
		else{
			opt = model.get(GRB.DoubleAttr.ObjVal);
			gap = model.get(GRB.DoubleAttr.MIPGap);
			System.out.println("OPTIMAL VALUE: "+model.get(GRB.DoubleAttr.ObjVal));
			for (int i = 0; i < NNs[0].numInputs; i++) {
				System.out.print(x[i].get(GRB.DoubleAttr.X)+" ");
			}
			System.out.println();
			/*for (int i = 0; i < numNN; i++) {
				for (int u = 0; u < NNs[i].numNodes; u++) {
					System.out.println("F"+u+": "+F[i].get(u).get(GRB.DoubleAttr.X));
					System.out.println("G"+u+": "+G[i].get(u).get(GRB.DoubleAttr.X));
					System.out.println("x"+u+": "+x[i].get(u).get(GRB.DoubleAttr.X));
				}
			}
			 */
		}
	}

	// Branch and cut from the MP paper
	public void IPBC() throws GRBException {
		env.set(GRB.IntParam.OutputFlag, 1);
		//env.set(GRB.IntParam.Cuts, 0);
		//env.set(GRB.IntParam.Presolve, 0);
		model = new GRBModel(env);
		double timeUsed = (System.currentTimeMillis()-startTime)/1000;
		model.set(GRB.DoubleParam.TimeLimit, 3600.0-timeUsed);
		if(isMax) {model.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // most y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(model.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(model.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, ""));}
					h[i].add(model.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					if(NNs[i].layers[l].get(v).fixActivation==9) {z[i].add(model.addVar(0, 1, 0, GRB.BINARY, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {z[i].add(model.addVar(0, 0, 0, GRB.CONTINUOUS, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {z[i].add(model.addVar(1, 1, 0, GRB.CONTINUOUS, ""));}
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = model.addVar(-1, 1, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		model.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// h_v=y_v = x_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				model.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				model.addConstr(expr2, GRB.EQUAL, 0, "");
			}	
			// y_t = h_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			model.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					model.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					if(NNs[i].layers[l].get(v).fixActivation==9) { // Binary variable not fixed
						double lb = -NNs[i].layers[l].get(v).lb;
						double ub = NNs[i].layers[l].get(v).ub;
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-ub, z[i].get(vindex));
						model.addConstr(expr, GRB.LESS_EQUAL, 0, "");

						GRBLinExpr expr6 = new GRBLinExpr();
						expr6.addTerm(1, y[i].get(vindex));
						model.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr2 = new GRBLinExpr();
						expr2.addTerm(1, y[i].get(vindex));
						expr2.addTerm(-1, h[i].get(vindex));
						model.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr4 = new GRBLinExpr();
						expr4.addTerm(1, y[i].get(vindex));
						expr4.addTerm(-1, h[i].get(vindex));
						expr4.addTerm(lb, z[i].get(vindex));
						model.addConstr(expr4, GRB.LESS_EQUAL, lb, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {//Node always inactive 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						model.addConstr(expr, GRB.EQUAL, 0, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {//Node always active 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-1, h[i].get(vindex));
						model.addConstr(expr, GRB.EQUAL, 0, "");
					}
				}
			}

		}

		// Solve
		model.update();
		model.setCallback(new CallbackBC_MP(this , y, z));
		model.optimize();
		if(model.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			System.out.println("		MIP is infeasible!");		
		}
		else{
			opt = model.get(GRB.DoubleAttr.ObjVal);
			gap = model.get(GRB.DoubleAttr.MIPGap);
			System.out.println("OPTIMAL VALUE: "+model.get(GRB.DoubleAttr.ObjVal));
			/*for (int i = 0; i < NNs[0].numInputs; i++) {
				System.out.print(x[i].get(GRB.DoubleAttr.X)+" ");
			}
			System.out.println();
			System.out.println("TOTAL CUTS: "+totalCuts);
			/*for (int i = 0; i < numNN; i++) {
				for (int u = 0; u < NNs[i].numNodes; u++) {
					System.out.println("F"+u+": "+F[i].get(u).get(GRB.DoubleAttr.X));
					System.out.println("G"+u+": "+G[i].get(u).get(GRB.DoubleAttr.X));
					System.out.println("x"+u+": "+x[i].get(u).get(GRB.DoubleAttr.X));
				}
			}
			 */
		}
	}

	public void survey() throws GRBException {
		env.set(GRB.IntParam.OutputFlag, 0);
		model = new GRBModel(env);
		model.set(GRB.DoubleParam.TimeLimit, 3600.0);
		if(isMax) {model.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // most y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(model.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(model.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, ""));}
					h[i].add(model.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					if(NNs[i].layers[l].get(v).fixActivation==9) {z[i].add(model.addVar(0, 1, 0, GRB.BINARY, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {z[i].add(model.addVar(0, 0, 0, GRB.CONTINUOUS, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {z[i].add(model.addVar(1, 1, 0, GRB.CONTINUOUS, ""));}
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = model.addVar(-1, 1, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		model.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// h_v=y_v = x_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				model.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				model.addConstr(expr2, GRB.EQUAL, 0, "");
			}	
			// y_t = h_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			model.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					model.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					if(NNs[i].layers[l].get(v).fixActivation==9) { // Binary variable not fixed
						double lb = -NNs[i].layers[l].get(v).lb;
						double ub = NNs[i].layers[l].get(v).ub;
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-ub, z[i].get(vindex));
						model.addConstr(expr, GRB.LESS_EQUAL, 0, "");

						GRBLinExpr expr6 = new GRBLinExpr();
						expr6.addTerm(1, y[i].get(vindex));
						model.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr2 = new GRBLinExpr();
						expr2.addTerm(1, y[i].get(vindex));
						expr2.addTerm(-1, h[i].get(vindex));
						model.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr4 = new GRBLinExpr();
						expr4.addTerm(1, y[i].get(vindex));
						expr4.addTerm(-1, h[i].get(vindex));
						expr4.addTerm(lb, z[i].get(vindex));
						model.addConstr(expr4, GRB.LESS_EQUAL, lb, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {//Node always inactive 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						model.addConstr(expr, GRB.EQUAL, 0, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {//Node always active 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-1, h[i].get(vindex));
						model.addConstr(expr, GRB.EQUAL, 0, "");
					}
				}
			}

		}

		// Solve
		model.update();
		model.setCallback(new CallbackSurvey(this , y, h));
		model.optimize();
		if(model.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			System.out.println("		SURVEY is infeasible!");		
		}
		else{
			System.out.println("DONE SURVEY!");
			totalCuts = 0;
			//for (int i = 0; i < numNN; i++) {
			//	NNs[i].printBounds();
			//}
			model.reset();
			model.dispose();
			/*for (int i = 0; i < numNN; i++) {
				for (int u = 0; u < NNs[i].numNodes; u++) {
					System.out.println("F"+u+": "+F[i].get(u).get(GRB.DoubleAttr.X));
					System.out.println("G"+u+": "+G[i].get(u).get(GRB.DoubleAttr.X));
					System.out.println("x"+u+": "+x[i].get(u).get(GRB.DoubleAttr.X));
				}
			}
			 */
		}
	}



	public void boundIP(double varlb, double varub) throws GRBException {
		env.set(GRB.IntParam.OutputFlag, 0);
		modelB = new GRBModel(env);
		modelB.set(GRB.DoubleParam.TimeLimit, 5.0);

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					y[i].add(modelB.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					h[i].add(modelB.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					z[i].add(modelB.addVar(0, 1, 0, GRB.BINARY, ""));
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = modelB.addVar(varlb, varub, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		modelB.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// x_v=h_v = y_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				modelB.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				modelB.addConstr(expr2, GRB.EQUAL, 0, "");
				NNs[i].layers[0].get(v).lb = varlb; //Input nodes have same bounds as input variables
				NNs[i].layers[0].get(v).ub = varub;
			}	

			// G_t = F_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			modelB.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					modelB.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				//Solve two bounding problems to compute lb and ub BEFORE adding the constraints on y 
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					h[i].get(vindex).set(GRB.DoubleAttr.Obj, 1); // to find lb
					modelB.update();
					modelB.optimize(); numCritical++;
					if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
						NNs[i].layers[l].get(v).lb = modelB.get(GRB.DoubleAttr.ObjVal)-tol;
					}
					else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
						NNs[i].layers[l].get(v).lb = modelB.get(GRB.DoubleAttr.ObjBound)-tol;
						System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
					}
					h[i].get(vindex).set(GRB.DoubleAttr.Obj, -1); // to find ub
					modelB.update();
					modelB.optimize();
					if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
						NNs[i].layers[l].get(v).ub = -1*modelB.get(GRB.DoubleAttr.ObjVal)+tol;
					}
					else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
						NNs[i].layers[l].get(v).ub = -1*modelB.get(GRB.DoubleAttr.ObjBound)+tol;
						System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
					}
					h[i].get(vindex).set(GRB.DoubleAttr.Obj, 0); // back to 0

					// Check if binary can be fixed
					if(NNs[i].layers[l].get(v).lb >=0 ) {NNs[i].layers[l].get(v).fixActivation = 1; z[i].get(vindex).set(GRB.DoubleAttr.LB, 1);}
					if(NNs[i].layers[l].get(v).ub <=0 ) {NNs[i].layers[l].get(v).fixActivation = 0; z[i].get(vindex).set(GRB.DoubleAttr.UB, 0);}
				}				

				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					if(NNs[i].layers[l].get(v).fixActivation==9) { // Binary variable not fixed
						double lb = -NNs[i].layers[l].get(v).lb;
						double ub = NNs[i].layers[l].get(v).ub;
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-ub, z[i].get(vindex));
						modelB.addConstr(expr, GRB.LESS_EQUAL, 0, "");

						GRBLinExpr expr6 = new GRBLinExpr();
						expr6.addTerm(1, y[i].get(vindex));
						modelB.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr2 = new GRBLinExpr();
						expr2.addTerm(1, y[i].get(vindex));
						expr2.addTerm(-1, h[i].get(vindex));
						modelB.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr4 = new GRBLinExpr();
						expr4.addTerm(1, y[i].get(vindex));
						expr4.addTerm(-1, h[i].get(vindex));
						expr4.addTerm(lb, z[i].get(vindex));
						modelB.addConstr(expr4, GRB.LESS_EQUAL, lb, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {//Node always inactive 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						modelB.addConstr(expr, GRB.EQUAL, 0, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {//Node always active 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-1, h[i].get(vindex));
						modelB.addConstr(expr, GRB.EQUAL, 0, "");
					}
				}
			}

			//Compute bounds for last node

			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, 1); // to find lb
			modelB.update();
			modelB.optimize(); numCritical++;
			if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
				NNs[i].layers[NNs[i].numLayers-1].get(0).lb = modelB.get(GRB.DoubleAttr.ObjVal)-tol;
			}
			else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
				NNs[i].layers[NNs[i].numLayers-1].get(0).lb = modelB.get(GRB.DoubleAttr.ObjBound)-tol;
				System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
			}

			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, -1); // to find ub
			modelB.update();
			modelB.optimize();
			if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
				NNs[i].layers[NNs[i].numLayers-1].get(0).ub = -1*modelB.get(GRB.DoubleAttr.ObjVal)+tol;
			}
			else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
				NNs[i].layers[NNs[i].numLayers-1].get(0).ub = -1*modelB.get(GRB.DoubleAttr.ObjBound)+tol;
				System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
			}
			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, 0); // back to 0

			//NNs[i].printBounds();
			System.out.println("Done bounds for NN "+i);
		}

	}

	public void boundTargeted(double varlb, double varub) throws GRBException {
		envB.set(GRB.IntParam.OutputFlag, 0);
		modelB = new GRBModel(envB);
		modelB.set(GRB.DoubleParam.TimeLimit, 5.0);

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // y variables are nonnegative!!!

					y[i].add(modelB.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));
					h[i].add(modelB.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					z[i].add(modelB.addVar(0, 1, 0, GRB.BINARY, ""));
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = modelB.addVar(varlb, varub, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		modelB.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// x_v=h_v = y_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				modelB.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				modelB.addConstr(expr2, GRB.EQUAL, 0, "");
				NNs[i].layers[0].get(v).lb = varlb; //Input nodes have same bounds as input variables
				NNs[i].layers[0].get(v).ub = varub;
			}	

			// G_t = F_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			modelB.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					modelB.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {

				//Solve two bounding problems to compute lb and ub BEFORE adding the constraints on y 
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					if(NNs[i].layers[l].get(v).violations/NNs[i].layers[l].get(v).numCritical >= 0.01) {
						//System.out.println("OLD BOUNDS: ");
						//NNs[i].layers[l].get(v).printBounds();


						lpbounds += (NNs[i].layers[l].get(v).ub - NNs[i].layers[l].get(v).lb);

						int vindex = NNs[i].layers[l].get(v).ID;
						h[i].get(vindex).set(GRB.DoubleAttr.Obj, 1); // to find lb
						modelB.update();
						modelB.optimize(); numCritical++;
						if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
							NNs[i].layers[l].get(v).lb = modelB.get(GRB.DoubleAttr.ObjVal)-tol;
						}
						else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
							NNs[i].layers[l].get(v).lb = modelB.get(GRB.DoubleAttr.ObjBound)-tol;
							System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
						}
						h[i].get(vindex).set(GRB.DoubleAttr.Obj, -1); // to find ub
						modelB.update();
						modelB.optimize();
						if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
							NNs[i].layers[l].get(v).ub = -1*modelB.get(GRB.DoubleAttr.ObjVal)+tol;
						}
						else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
							NNs[i].layers[l].get(v).ub = -1*modelB.get(GRB.DoubleAttr.ObjBound)+tol;
							System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
						}
						h[i].get(vindex).set(GRB.DoubleAttr.Obj, 0); // back to 0

						// Check if binary can be fixed
						if(NNs[i].layers[l].get(v).lb >=0 ) {NNs[i].layers[l].get(v).fixActivation = 1; z[i].get(vindex).set(GRB.DoubleAttr.LB, 1);}
						if(NNs[i].layers[l].get(v).ub <=0 ) {NNs[i].layers[l].get(v).fixActivation = 0; z[i].get(vindex).set(GRB.DoubleAttr.UB, 0);}

						//System.out.println("NEW BOUNDS: ");
						//NNs[i].layers[l].get(v).printBounds();
						ipbounds += (NNs[i].layers[l].get(v).ub - NNs[i].layers[l].get(v).lb);
						//System.out.println(lpbounds+"   "+ipbounds+"  "+numCritical);
					}
				}				

				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					if(NNs[i].layers[l].get(v).fixActivation==9) { // Binary variable not fixed
						double lb = -NNs[i].layers[l].get(v).lb;
						double ub = NNs[i].layers[l].get(v).ub;
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-ub, z[i].get(vindex));
						modelB.addConstr(expr, GRB.LESS_EQUAL, 0, "");

						GRBLinExpr expr6 = new GRBLinExpr();
						expr6.addTerm(1, y[i].get(vindex));
						modelB.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr2 = new GRBLinExpr();
						expr2.addTerm(1, y[i].get(vindex));
						expr2.addTerm(-1, h[i].get(vindex));
						modelB.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr4 = new GRBLinExpr();
						expr4.addTerm(1, y[i].get(vindex));
						expr4.addTerm(-1, h[i].get(vindex));
						expr4.addTerm(lb, z[i].get(vindex));
						modelB.addConstr(expr4, GRB.LESS_EQUAL, lb, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {//Node always inactive 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						modelB.addConstr(expr, GRB.EQUAL, 0, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {//Node always active 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-1, h[i].get(vindex));
						modelB.addConstr(expr, GRB.EQUAL, 0, "");
					}
				}
			}

			//Compute bounds for last node

			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, 1); // to find lb
			modelB.update();
			modelB.optimize(); numCritical++;
			if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
				NNs[i].layers[NNs[i].numLayers-1].get(0).lb = modelB.get(GRB.DoubleAttr.ObjVal)-tol;
			}
			else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
				NNs[i].layers[NNs[i].numLayers-1].get(0).lb = modelB.get(GRB.DoubleAttr.ObjBound)-tol;
				System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
			}

			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, -1); // to find ub
			modelB.update();
			modelB.optimize();
			if(modelB.get(GRB.IntAttr.Status) == GRB.OPTIMAL){
				NNs[i].layers[NNs[i].numLayers-1].get(0).ub = -1*modelB.get(GRB.DoubleAttr.ObjVal)+tol;
			}
			else if(modelB.get(GRB.IntAttr.Status) == GRB.TIME_LIMIT) {
				NNs[i].layers[NNs[i].numLayers-1].get(0).ub = -1*modelB.get(GRB.DoubleAttr.ObjBound)+tol;
				System.out.println("***********************Gap: "+modelB.get(GRB.DoubleAttr.MIPGap));
			}
			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, 0); // back to 0

			//NNs[i].printBounds();
			System.out.println("Done bounds for NN "+i);
		}


	}


	public void boundLP(double varlb, double varub) throws GRBException {
		envS.set(GRB.IntParam.OutputFlag, 0);
		modelS = new GRBModel(envS);

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					y[i].add(modelS.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					h[i].add(modelS.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					z[i].add(modelS.addVar(0, 1, 0, GRB.CONTINUOUS, ""));
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = modelS.addVar(varlb, varub, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		modelS.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// x_v=h_v = y_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				modelS.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				modelS.addConstr(expr2, GRB.EQUAL, 0, "");
				NNs[i].layers[0].get(v).lb = varlb; //Input nodes have same bounds as input variables
				NNs[i].layers[0].get(v).ub = varub;
			}	

			// G_t = F_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			modelS.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					modelS.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				//Solve two bounding problems to compute lb and ub BEFORE adding the constraints on y 
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					h[i].get(vindex).set(GRB.DoubleAttr.Obj, 1); // to find lb
					modelS.update();
					modelS.optimize();
					NNs[i].layers[l].get(v).lb = modelS.get(GRB.DoubleAttr.ObjVal)-tol;
					h[i].get(vindex).set(GRB.DoubleAttr.Obj, -1); // to find ub
					modelS.update();
					modelS.optimize();
					NNs[i].layers[l].get(v).ub = -1*modelS.get(GRB.DoubleAttr.ObjVal)+tol;
					h[i].get(vindex).set(GRB.DoubleAttr.Obj, 0); // back to 0

					// Check if binary can be fixed
					if(NNs[i].layers[l].get(v).lb >=0 ) {NNs[i].layers[l].get(v).fixActivation = 1;}
					if(NNs[i].layers[l].get(v).ub <=0 ) {NNs[i].layers[l].get(v).fixActivation = 0;}
					if(NNs[i].layers[l].get(v).ub-NNs[i].layers[l].get(v).lb > maxDom) {maxDom = NNs[i].layers[l].get(v).ub-NNs[i].layers[l].get(v).lb;}

				}				

				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					if(NNs[i].layers[l].get(v).fixActivation==9) { // Binary variable not fixed
						double lb = -NNs[i].layers[l].get(v).lb;
						double ub = NNs[i].layers[l].get(v).ub;
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-ub, z[i].get(vindex));
						modelS.addConstr(expr, GRB.LESS_EQUAL, 0, "");

						GRBLinExpr expr6 = new GRBLinExpr();
						expr6.addTerm(1, y[i].get(vindex));
						modelS.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr2 = new GRBLinExpr();
						expr2.addTerm(1, y[i].get(vindex));
						expr2.addTerm(-1, h[i].get(vindex));
						modelS.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

						GRBLinExpr expr4 = new GRBLinExpr();
						expr4.addTerm(1, y[i].get(vindex));
						expr4.addTerm(-1, h[i].get(vindex));
						expr4.addTerm(lb, z[i].get(vindex));
						modelS.addConstr(expr4, GRB.LESS_EQUAL, lb, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {//Node always inactive 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						modelS.addConstr(expr, GRB.EQUAL, 0, "");
					}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {//Node always active 
						GRBLinExpr expr = new GRBLinExpr();
						expr.addTerm(1, y[i].get(vindex));
						expr.addTerm(-1, h[i].get(vindex));
						modelS.addConstr(expr, GRB.EQUAL, 0, "");
					}
				}
			}

			//Compute bounds for last node
			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, 1); // to find lb
			modelS.update();
			modelS.optimize();
			NNs[i].layers[NNs[i].numLayers-1].get(0).lb = modelS.get(GRB.DoubleAttr.ObjVal)-tol;
			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, -1); // to find ub
			modelS.update();
			modelS.optimize();
			NNs[i].layers[NNs[i].numLayers-1].get(0).ub = -1*modelS.get(GRB.DoubleAttr.ObjVal)+tol;
			h[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, 0); // back to 0

			//NNs[i].printBounds();
			System.out.println("LP Bounds done NN "+i);
		}
	}




	private void print(ArrayList<double[]> sol) {
		for (int i = 0; i < sol.size(); i++) {
			System.out.print("SOL "+i+": ");
			for (int j = 0; j < sol.get(i).length; j++) {
				System.out.print(sol.get(i)[j]+" ");
			}
		}
		System.out.println();

	}

	// Computes the Lhat and Uhat constants for every node in every NN
	public void getCoeff() {
		for (int i = 0; i < numNN; i++) {
			NNs[i].computeCoeff();
			System.out.println("DONE "+i);
		}

	}

	public bendersCut solveSubproblem() throws GRBException {
		//for (int i = 0; i < numNN; i++) {
		//	System.out.println(zval[i]);
		//}
		envS.set(GRB.IntParam.OutputFlag, 0);
		modelS = new GRBModel(envS);
		if(isMax) {modelS.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBConstr> [] constraintsA = new ArrayList[numNN]; // 2e
		ArrayList<GRBConstr> [] constraintsB = new ArrayList[numNN]; // 2f
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			constraintsA[i] = new ArrayList<GRBConstr>();
			constraintsB[i] = new ArrayList<GRBConstr>();

			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(modelS.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(modelS.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, ""));}
					h[i].add(modelS.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = modelS.addVar(-1, 1, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		modelS.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// alpha_v=G_v = F_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				modelS.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				modelS.addConstr(expr2, GRB.EQUAL, 0, "");
			}	
			// G_t = F_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			modelS.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					modelS.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;

					double lb = -NNs[i].layers[l].get(v).lb;
					double ub = NNs[i].layers[l].get(v).ub;

					GRBLinExpr expr6 = new GRBLinExpr();
					expr6.addTerm(1, y[i].get(vindex));
					modelS.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

					GRBLinExpr expr2 = new GRBLinExpr();
					expr2.addTerm(1, y[i].get(vindex));
					expr2.addTerm(-1, h[i].get(vindex));
					modelS.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

					GRBLinExpr expr4 = new GRBLinExpr();
					expr4.addTerm(1, y[i].get(vindex));
					expr4.addTerm(-1, h[i].get(vindex));
					constraintsA[i].add( modelS.addConstr(expr4, GRB.LESS_EQUAL, lb*(1-zval[i].get(vindex)), "") );

					GRBLinExpr expr = new GRBLinExpr();
					expr.addTerm(1, y[i].get(vindex));
					constraintsB[i].add( modelS.addConstr(expr, GRB.LESS_EQUAL, ub*zval[i].get(vindex), "") );

				}
			}

		}

		// Solve
		modelS.update();
		modelS.optimize();
		if(modelS.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			System.out.println("		Benders Subproblem is infeasible!");
			//findINFCut();
			return null;//findINFCut();
		}
		else{
			//System.out.println("WEPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
			opt = modelS.get(GRB.DoubleAttr.ObjVal);
			System.out.println("Subproblem OBJ: "+modelS.get(GRB.DoubleAttr.ObjVal));
			if(isMax==false && opt < UB) {
				UB = opt;
				//	System.out.println("NEW UB: "+UB);
			}
			if(isMax==true && opt > LB) {
				LB = opt;
			}

			bendersCut cut = new bendersCut(numNN, 0, "OPT");

			for (int i = 0; i < numNN; i++) {
				int cindex = 0;
				for (int l = 1; l < NNs[i].numLayers-1; l++) {
					for (int v = 0; v < NNs[i].layers[l].size(); v++) {
						int vindex = NNs[i].layers[l].get(v).ID;
						double lb = -NNs[i].layers[l].get(v).lb; //New bounds once z changes
						double ub = NNs[i].layers[l].get(v).ub;

						double dualA = constraintsA[i].get(cindex).get(GRB.DoubleAttr.Pi);
						double dualB = constraintsB[i].get(cindex).get(GRB.DoubleAttr.Pi);
						//System.out.println(vindex+":   "+dualA+" "+dualB+ "  zval: "+zval[i].get(vindex));
						if(zval[i].get(vindex) == 0) {
							double c = dualB*ub-dualA*lb;
							cut.coef0[i].add(c);
							cut.coef1[i].add(0.0);
							//System.out.println("     Coeff0: "+c);
						}
						else {
							double c = dualA*lb-dualB*ub;
							cut.coef0[i].add(0.0);
							cut.coef1[i].add(c);
							//System.out.println("     Coeff1: "+c);
						}

						//System.out.println("var "+vindex+" coeffs: "+cut.coef0[i].get(cindex)+"    "+cut.coef1[i].get(cindex));
						cindex++;

					}
				}
			}

			cut.constant = opt;
			//cut.print();
			cutPool.add(cut);
			checkCut(cut);
			return cut;

		}


	}


	private bendersCut findINFCut() throws GRBException {
		envS.set(GRB.IntParam.OutputFlag, 0);
		modelS = new GRBModel(envS);

		ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
		ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
		ArrayList<GRBConstr> [] constraintsA = new ArrayList[numNN]; // 2e
		ArrayList<GRBConstr> [] constraintsB = new ArrayList[numNN]; // 2f
		ArrayList<GRBVar> [] sA = new ArrayList[numNN];
		ArrayList<GRBVar> [] sB = new ArrayList[numNN];
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		// Create variables
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			sA[i] = new ArrayList<GRBVar>();
			sB[i] = new ArrayList<GRBVar>();
			constraintsA[i] = new ArrayList<GRBConstr>();
			constraintsB[i] = new ArrayList<GRBConstr>();

			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(modelS.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(modelS.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					h[i].add(modelS.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					sA[i].add(modelS.addVar(0, 999999, 1, GRB.CONTINUOUS, ""));
					sB[i].add(modelS.addVar(0, 999999, 1, GRB.CONTINUOUS, ""));
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = modelS.addVar(-1, 1, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		modelS.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// alpha_v=G_v = F_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				modelS.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v]);
				modelS.addConstr(expr2, GRB.EQUAL, 0, "");
			}	
			// G_t = F_t for the last layer
			GRBLinExpr exprt = new GRBLinExpr();
			exprt.addTerm(1, y[i].get(NNs[i].numNodes-1));
			exprt.addTerm(-1, h[i].get(NNs[i].numNodes-1));
			modelS.addConstr(exprt, GRB.EQUAL, 0, "");
			// For every layer except the first one h_v=sum weigths y_u + Bias_v
			for (int l = 1; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					GRBLinExpr expr = new GRBLinExpr();
					int vindex = NNs[i].layers[l].get(v).ID;
					expr.addTerm(1, h[i].get(vindex));
					for (int u = 0; u < NNs[i].layers[l-1].size(); u++) {
						int uindex = NNs[i].layers[l-1].get(u).ID;
						expr.addTerm(-NNs[i].layers[l].get(v).weights[u], y[i].get(uindex));
					}
					modelS.addConstr(expr, GRB.EQUAL, NNs[i].layers[l].get(v).bias, "");
				}
			}
			// For every layer except the first and last F_v=max{0, G_v}
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;

					double lb = -NNs[i].layers[l].get(v).lb;
					double ub = NNs[i].layers[l].get(v).ub;

					GRBLinExpr expr6 = new GRBLinExpr();
					expr6.addTerm(1, y[i].get(vindex));
					modelS.addConstr(expr6, GRB.GREATER_EQUAL, 0, "");

					GRBLinExpr expr2 = new GRBLinExpr();
					expr2.addTerm(1, y[i].get(vindex));
					expr2.addTerm(-1, h[i].get(vindex));
					modelS.addConstr(expr2, GRB.GREATER_EQUAL, 0, "");

					GRBLinExpr expr4 = new GRBLinExpr();
					expr4.addTerm(1, y[i].get(vindex));
					expr4.addTerm(-1, h[i].get(vindex));
					expr4.addTerm(-1, sA[i].get(vindex));
					constraintsA[i].add( modelS.addConstr(expr4, GRB.LESS_EQUAL, lb*(1-zval[i].get(vindex)), "") );

					GRBLinExpr expr = new GRBLinExpr();
					expr.addTerm(1, y[i].get(vindex));
					expr.addTerm(-1, sB[i].get(vindex));
					constraintsB[i].add( modelS.addConstr(expr, GRB.LESS_EQUAL, ub*zval[i].get(vindex), "") );

				}
			}

		}

		// Solve
		modelS.update();
		modelS.optimize();
		if(modelS.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			System.out.println("		findINF is infeasible!");		
			return null;
		}
		else{
			//System.out.println("WEPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
			opt = modelS.get(GRB.DoubleAttr.ObjVal);
			System.out.println("INF OBJ: "+modelS.get(GRB.DoubleAttr.ObjVal));
			bendersCut cut = new bendersCut(numNN, 0, "INF");

			for (int i = 0; i < numNN; i++) {
				int cindex = 0;
				for (int l = 1; l < NNs[i].numLayers-1; l++) {
					for (int v = 0; v < NNs[i].layers[l].size(); v++) {
						int vindex = NNs[i].layers[l].get(v).ID;
						double lb = -NNs[i].layers[l].get(v).lb;
						double ub = NNs[i].layers[l].get(v).ub;

						double dualA = constraintsA[i].get(cindex).get(GRB.DoubleAttr.Pi);
						double dualB = constraintsB[i].get(cindex).get(GRB.DoubleAttr.Pi);
						//System.out.println(vindex+":   "+dualA+" "+dualB+ "  zval: "+zval[i].get(vindex));
						if(zval[i].get(vindex) == 0) {
							double c = dualB*ub-dualA*lb;
							cut.coef0[i].add(c);
							cut.coef1[i].add(0.0);
							//System.out.println("     Coeff0: "+c);
						}
						else {
							double c = dualA*lb-dualB*ub;
							cut.coef0[i].add(0.0);
							cut.coef1[i].add(c);
							//System.out.println("     Coeff1: "+c);
						}

						//System.out.println("var "+vindex+" coeffs: "+cut.coef0[i].get(cindex)+"    "+cut.coef1[i].get(cindex));
						cindex++;

					}
				}
			}

			cut.constant = opt;
			//cut.print();
			cutPool.add(cut);
			//checkCut(cut);
			return cut;

		}



	}

	private void checkCut(bendersCut cut) {
		// Check cut with current zval
		double sum = 0;
		for (int i = 0; i < numNN; i++) {
			int cindex = 0;
			for (int l = 1; l < NNs[i].numLayers-1; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;
					sum += cut.coef0[i].get(cindex)*zval[i].get(vindex);
					sum += cut.coef1[i].get(cindex)*( 1-zval[i].get(vindex) );
					//System.out.println("value of var "+vindex+": "+zval[i].get(vindex)+" coef0: "+cut.coef0[i].get(cindex)+" coef1: "+cut.coef1[i].get(cindex));
					cindex++;
				}
			}
		}
		System.out.println("VALUE OF CUT WITH CURRENT Z: "+sum);

	}

	public void cleanModels() throws GRBException {
		model.reset();
		model.dispose();
		//modelS.reset();
		//modelS.dispose();
		
		env.dispose();
		//envS.dispose();
		envB.dispose();
	}





}
