package twoPhaseSensitivity;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import gurobi.*;

public class algHandler {

	// Number of trees
	int numNN;
	boolean isMax;

	// Problem Data
	NN[] NNs;						//All the NNs 										 

	double primalBound;				// Obj of a feasible solution

	GRBEnv env;						// Gurobi Environment
	GRBModel model;					// MIPmodel for LR
	double opt;						// Optimal solution
	double gap;						// MIP Gap
	double tol = 0.00001;			// Numerical tolerance
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
	boolean addBenders;

	// For stats
	double lpbounds;					// Average width of the bounds via LPs
	double ipbounds;					// Average width of the bounds via IPs
	double numCritical;					// Number of critical neurons

	// For BB
	double[] xval;
	ArrayList<BBNode> tree;
	double[][] lambda;					// Lagrangian Multipliers 	
	double epsilon = 0.02;				// Threshold for variables to differ
	double stepSize = 0.1;
	Boolean solFound;

	// Stats 
	double timeBounding;
	double timePhase1;
	double timePhase2;
	double timeGradient;
	double bound1;
	double bound2;
	int numberNodesExplored;
	int numBrute;
	int Depth;
		
	public algHandler() throws GRBException {
		env = new GRBEnv(null);
		envS = new GRBEnv(null);
		envB = new GRBEnv(null);
		isMax = false;
		maxDom = 0;
		UB= 999999;
		LB = -999999;
		addBenders = false;
		timeBounding = 0;
		timePhase1 = 0;
		timePhase2 = 0;
		numberNodesExplored = 0;
		timeGradient = 0;
		numBrute = 0;
		Depth = 0;
		bound1 = 0;
		bound2 = 0;
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

	public void survey() throws GRBException {
		envS.set(GRB.IntParam.OutputFlag, 0);
		modelS = new GRBModel(envS);
		modelS.set(GRB.DoubleParam.TimeLimit, 3600.0);
		if(isMax) {modelS.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

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
					if(l < NNs[i].numLayers-1) {y[i].add(modelS.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(modelS.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, ""));}
					h[i].add(modelS.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					if(NNs[i].layers[l].get(v).fixActivation==9) {z[i].add(modelS.addVar(0, 1, 0, GRB.BINARY, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {z[i].add(modelS.addVar(0, 0, 0, GRB.CONTINUOUS, ""));}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {z[i].add(modelS.addVar(1, 1, 0, GRB.CONTINUOUS, ""));}
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
			// h_v=y_v = x_v for the first layer
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
			// y_t = h_t for the last layer
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

		}

		// Solve
		modelS.update();
		modelS.setCallback(new CallbackSurvey(this , y, h));
		modelS.optimize();
		if(modelS.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			System.out.println("		SURVEY is infeasible!");		
		}
		else{
			System.out.println("DONE SURVEY!");
			totalCuts = 0;
			//for (int i = 0; i < numNN; i++) {
			//	NNs[i].printBounds();
			//}
			modelS.reset();
			modelS.dispose();
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



	public void boundFest(double varlb, double varub) throws GRBException {
		env.set(GRB.IntParam.OutputFlag, 0);
		modelB = new GRBModel(env);
		modelB.set(GRB.DoubleParam.TimeLimit, 10.0);

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
					modelB.optimize();
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
			modelB.optimize();
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


		/*	// Solve final problem
		double timeUsed = (System.currentTimeMillis()-startTime)/1000;
		model.set(GRB.DoubleParam.TimeLimit, 3600.0-timeUsed);	
		for (int i = 0; i < numNN; i++) {
			y[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, -1); // to find lb
		}
		model.update();
		model.optimize();
		opt = model.get(GRB.DoubleAttr.ObjVal);
		System.out.println("OPTIMAL VALUE: "+model.get(GRB.DoubleAttr.ObjVal)/numNN);
		for (int i = 0; i < NNs[0].numInputs; i++) {
			System.out.print(x[i].get(GRB.DoubleAttr.X)+" ");
		}
		System.out.println(); 
		 */
	}

	public void boundFestTargeted(double varlb, double varub) throws GRBException {
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

						numCritical++;
						lpbounds += (NNs[i].layers[l].get(v).ub - NNs[i].layers[l].get(v).lb);

						int vindex = NNs[i].layers[l].get(v).ID;
						h[i].get(vindex).set(GRB.DoubleAttr.Obj, 1); // to find lb
						modelB.update();
						modelB.optimize();
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
			modelB.optimize();
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


		/*	// Solve final problem
		double timeUsed = (System.currentTimeMillis()-startTime)/1000;
		model.set(GRB.DoubleParam.TimeLimit, 3600.0-timeUsed);	
		for (int i = 0; i < numNN; i++) {
			y[i].get(NNs[i].numNodes-1).set(GRB.DoubleAttr.Obj, -1); // to find lb
		}
		model.update();
		model.optimize();
		opt = model.get(GRB.DoubleAttr.ObjVal);
		System.out.println("OPTIMAL VALUE: "+model.get(GRB.DoubleAttr.ObjVal)/numNN);
		for (int i = 0; i < NNs[0].numInputs; i++) {
			System.out.print(x[i].get(GRB.DoubleAttr.X)+" ");
		}
		System.out.println(); 
		 */
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

	public void primalHeuristic() throws GRBException {
		//Set cutoff
		if(!isMax) {modelS.set(GRB.DoubleParam.Cutoff, UB);}
		if(isMax) {modelS.set(GRB.DoubleParam.Cutoff, LB);}
		
		GRBVar [] x = new GRBVar[NNs[0].numInputs]; 
		for (int j = 0; j < x.length; j++) {
			x[j] = modelS.getVarByName("x_"+j);
		}
		//System.out.println("CURRENT X VAL: ");
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j].set(GRB.DoubleAttr.LB, Math.max(xval[j]-epsilon, -1)); 
			x[j].set(GRB.DoubleAttr.UB, Math.min(xval[j]+epsilon, 1));
			//System.out.print(xval[j]+" ");
		}
		//System.out.println();
		// Get binary variables
		/*ArrayList<GRBVar> [] z = new ArrayList[numNN];
		for (int i = 0; i < numNN; i++) {
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					z[i].add(modelS.getVarByName("z_"+i+","+l+","+v));
				}
			}
		}	
		 */

		// Solve
		modelS.update();
		modelS.optimize();
		if(modelS.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			/*System.out.println("		PRIMAL HEURISTIC is infeasible!");
			System.out.println("CURRENT X VAL: ");
			for (int j = 0; j < NNs[0].numInputs; j++) {
				System.out.print(xval[j]+" ");
			}
			System.out.println();
			System.out.println("VAR BOUNDS: ");
			for (int j = 0; j < NNs[0].numInputs; j++) {
				System.out.println(x[j].get(GRB.DoubleAttr.LB)+" , "+x[j].get(GRB.DoubleAttr.UB));
			}*/
		}
		else{
			opt = round(modelS.get(GRB.DoubleAttr.ObjVal));

			//System.out.println("PRIMAL HEURISTIC VALUE: "+modelS.get(GRB.DoubleAttr.ObjVal));
			//for (int i = 0; i < NNs[0].numInputs; i++) {
			//	System.out.print(x[i].get(GRB.DoubleAttr.X)+" ");
			//}
			//System.out.println();
			if(isMax==false && opt < UB) {
				UB = opt;
				//System.out.println("**************NEW UB: "+UB);
				addBenders = true;
				// RECORD ZVALUES FOR BENDERS CUT
			/*	for (int i = 0; i < numNN; i++) {
					zval[i].clear();
					for (int j = 0; j < z[i].size(); j++) {
						zval[i].add((int) z[i].get(j).get(GRB.DoubleAttr.X));
					}	
					//System.out.println(zval[i]);
				}
			 */
			}
			if(isMax==true && opt > LB) {
				LB = opt;
				//System.out.println("**************NEW LB: "+LB);
				addBenders = true;
				// RECORD ZVALUES FOR BENDERS CUT
			/*	for (int i = 0; i < numNN; i++) {
					zval[i].clear();
					for (int j = 0; j < z[i].size(); j++) {
						zval[i].add((int) z[i].get(j).get(GRB.DoubleAttr.X));
					}	
					//System.out.println(zval[i]);
				}
			 */
			}

		}
	}

	private void addBendersCut(bendersCut cut) throws GRBException {

		// Get variables
		ArrayList<GRBVar> [] z = new ArrayList[numNN];
		for (int i = 0; i < numNN; i++) {
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					z[i].add(model.getVarByName("z_"+i+","+l+","+v));
				}
			}
		}	

		ArrayList<GRBVar> [] y = new ArrayList[numNN];
		for (int i = 0; i < numNN; i++) {
			y[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					y[i].add(model.getVarByName("y_"+i+","+l+","+v));
				}
			}
		}		
		
		// ADD BENDERS CUT
		double rhs = cut.constant;
		if(cut.type == "OPT") {
			GRBLinExpr exprC = new GRBLinExpr();
			for (int i = 0; i < numNN; i++) {
				exprC.addTerm(-(1/(numNN+0.0)), y[i].get(NNs[i].numNodes-1));
			}
			for (int i = 0; i < numNN; i++) {
				int cindex = 0;
				for (int l = 1; l < NNs[i].numLayers-1; l++) {
					for (int v = 0; v < NNs[i].layers[l].size(); v++) {
						int vindex = NNs[i].layers[l].get(v).ID;
						exprC.addTerm(cut.coef0[i].get(cindex), z[i].get(vindex));
						exprC.addTerm(-cut.coef1[i].get(cindex), z[i].get(vindex));
						rhs += cut.coef1[i].get(cindex);
						cindex++;
					}
				}
			}
			if(isMax==false) {model.addConstr(exprC, GRB.LESS_EQUAL, -rhs+tol, "");}
			else {model.addConstr(exprC, GRB.GREATER_EQUAL, -rhs-tol, "");}
			model.update();
			System.out.println("BENDERS CUT ADDED");
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
		//System.out.println("VALUE OF CUT WITH CURRENT Z: "+sum);

	}
	public bendersCut solveBendersSubproblem() throws GRBException {
		//for (int i = 0; i < numNN; i++) {
		//	System.out.println(zval[i]);
		//}
		envB.set(GRB.IntParam.OutputFlag, 0);
		modelB = new GRBModel(envB);
		if(isMax) {modelB.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

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
					if(l < NNs[i].numLayers-1) {y[i].add(modelB.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(modelB.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, ""));}
					h[i].add(modelB.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = modelB.addVar(-1, 1, 0, GRB.CONTINUOUS, "");
		}

		//Integrate variables
		modelB.update();

		// Add constraints for every NN
		for (int i = 0; i < numNN; i++) {
			// alpha_v=G_v = F_v for the first layer
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
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					int vindex = NNs[i].layers[l].get(v).ID;

					double lb = -NNs[i].layers[l].get(v).lb;
					double ub = NNs[i].layers[l].get(v).ub;

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
					constraintsA[i].add( modelB.addConstr(expr4, GRB.LESS_EQUAL, lb*(1-zval[i].get(vindex)), "") );

					GRBLinExpr expr = new GRBLinExpr();
					expr.addTerm(1, y[i].get(vindex));
					constraintsB[i].add( modelB.addConstr(expr, GRB.LESS_EQUAL, ub*zval[i].get(vindex), "") );

				}
			}

		}

		// Solve
		modelB.update();
		modelB.optimize();
		if(modelB.get(GRB.IntAttr.Status) == GRB.INFEASIBLE){
			System.out.println("		BENDERS Subproblem is infeasible!");
			return null;//findINFCut();
		}
		else{
			//System.out.println("WEPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
			opt = modelB.get(GRB.DoubleAttr.ObjVal);
			//System.out.println("BENDERS Subproblem OBJ: "+modelB.get(GRB.DoubleAttr.ObjVal));
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
			//checkCut(cut);
			return cut;

		}


	}



	public void initializeBB(GRBVar[][] x) throws GRBException {
		double rootBound = 0;
		env = new GRBEnv(null);
		env.set(GRB.IntParam.OutputFlag, 0);
		model = new GRBModel(env);
		model.set(GRB.DoubleParam.TimeLimit, 180.0);
		if(isMax) {model.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

		for (int i = 0; i < numNN; i++) {

			ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
			ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
			ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
			// Create variables
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			z[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(model.addVar(lby, uby, 0, GRB.CONTINUOUS, "y_"+i+","+l+","+v));}
					else {y[i].add(model.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, "y_"+i+","+l+","+v));}
					h[i].add(model.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					if(NNs[i].layers[l].get(v).fixActivation==9) {z[i].add(model.addVar(0, 1, 0, GRB.BINARY, "z_"+i+","+l+","+v));}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {z[i].add(model.addVar(0, 0, 0, GRB.CONTINUOUS, "z_"+i+","+l+","+v));}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {z[i].add(model.addVar(1, 1, 0, GRB.CONTINUOUS, "z_"+i+","+l+","+v));}
				}
			}
			for (int j = 0; j < NNs[0].numInputs; j++) {
				if(i==0) {x[j][i] = model.addVar(-1, 1, lambda[j][i], GRB.CONTINUOUS, "");}
				else {x[j][i] = model.addVar(-1, 1, -lambda[j][i], GRB.CONTINUOUS, "");}
			}

			//Integrate variables
			model.update();

			// Add constraints for every NN
			// alpha_v=G_v = F_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				model.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v][i]);
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
		model.update();
		model.optimize();
		rootBound = round(model.get(GRB.DoubleAttr.ObjBound));
		bound2 = rootBound;
		// Create root node
		System.out.println("ROOT NODE BOUND: "+rootBound);
		double[] lb = new double[x.length];
		double[] ub = new double[x.length];
		double[] minx = new double[x.length];
		double[] maxx = new double[x.length];
		tree = new ArrayList<BBNode>();

		for (int j = 0; j < NNs[0].numInputs; j++) {
			lb[j] = -1;
			ub[j] = 1;
			minx[j] = 1;
			maxx[j] = -1;
		}
		BBNode n = new BBNode ( rootBound, lb, ub, 0);
		//n.print();

		// Evaluate the NN for each x vector
		for (int i = 0; i < numNN; i++) {
			xval = new double[NNs[0].numInputs]; 
			for (int j = 0; j < NNs[0].numInputs; j++) {
				xval[j] = round(x[j][i].get(GRB.DoubleAttr.X));
				if(xval[j]<minx[j]) {minx[j] = xval[j];}
				if(xval[j]>maxx[j]) {maxx[j] = xval[j];}
			}
			if(i==0) {primalHeuristic();}
		}

		// Create the first two branches
		double maxDif = 0;
		int branchVar = -1;
		for (int j = 0; j < NNs[0].numInputs; j++) {
			//System.out.println("min and max x: "+minx[j]+"  "+maxx[j]);
			if(maxx[j] - minx[j] > maxDif) {
				maxDif = maxx[j] - minx[j];
				branchVar = j;
			}
		}

		if(maxDif > tol) {
			//System.out.println("BRANCH ON VARIABLE "+branchVar+" maxdifference = "+maxDif);

			BBNode leftn = new BBNode(n.bound, n.xLB, n.xUB, n.numBranches);
			BBNode rightn = new BBNode(n.bound, n.xLB, n.xUB, n.numBranches);

			leftn.xUB[branchVar] = round( (maxx[branchVar] + minx[branchVar])/2.0 ); 
			rightn.xLB[branchVar] = round( (maxx[branchVar] + minx[branchVar])/2.0 ); 

			//leftn.print();
			//rightn.print();
			tree.add(leftn);
			tree.add(rightn);
			
			//Update lagrangian multipliers
			double [][] diff = new double[NNs[0].numInputs][numNN];  // To capture the disagreement between variables
			for (int j = 0; j < NNs[0].numInputs; j++) {
				lambda[j][0] = 0;
				for (int i = 0; i < numNN; i++) {
					diff[j][i] = round(x[j][0].get(GRB.DoubleAttr.X)-x[j][i].get(GRB.DoubleAttr.X));
					if(!isMax) {lambda[j][i] = round(lambda[j][i]+stepSize*diff[j][i]);}
					if(isMax) {lambda[j][i] = round(lambda[j][i]-stepSize*diff[j][i]);}
					lambda[j][0] += lambda[j][i];
				}
			}

			// Update variable coeffs
			for (int i = 0; i < numNN; i++) {
				for (int j = 0; j < NNs[0].numInputs; j++) {
					if(i==0) {x[j][i].set(GRB.DoubleAttr.Obj, lambda[j][i]);}
					else {x[j][i].set(GRB.DoubleAttr.Obj, -lambda[j][i]);}
				}
			}
			model.update(); 

		}
		else {
			UB = n.bound;
			LB = n.bound;
		}

		

	}



	public void BranchBound() throws GRBException {

		double timeUsed = 0;
		GRBVar [][] x = new GRBVar[NNs[0].numInputs][numNN];	// Variables to branch  
		
		initializePrimalHeuristic();
		//solveLagDual();
		initializeBB(x);
		int iter = 1;
		while(tree.size() > 0 && (UB-LB)/(Math.abs(UB)) > 0.001 && timeUsed < 3600-20) {
			BBNode n = tree.get(0);
			tree.remove(0);
			numberNodesExplored++;
			solveNode(n , x);
			if(tree.size()>0) {
				if(isMax == false ) {LB = Math.max(LB, tree.get(0).bound);}
				else {UB = Math.min(UB, tree.get(0).bound);}
			}
			else { 
				if(isMax == false ) {LB = UB;}
				else {UB = LB;}
			}
			iter++;
			timeUsed = (System.currentTimeMillis()-startTime)/1000;
			
			if(iter%10==0) {
				System.out.println("NODES EXPLORED "+iter+" LB = "+LB+" UB = "+UB+" GAP: "+round( (UB-LB)/(Math.abs(UB)) ) );
				System.out.println("TIME USED: "+timeUsed);
			}

		}

		if(isMax == false ) { opt = UB; gap = (UB-LB)/(Math.abs(LB));}
		else { opt = LB; gap = (UB-LB)/(Math.abs(UB));}

	}

	private void solveLagDual() throws GRBException {
		model = new GRBModel(env);
		env.set(GRB.IntParam.OutputFlag, 0);
		model.set(GRB.DoubleParam.TimeLimit, 25.0);
		if(isMax) {model.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

		GRBVar [][] x = new GRBVar[NNs[0].numInputs][numNN];

		for (int i = 0; i < numNN; i++) {

			ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
			ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
			ArrayList<GRBVar> [] z = new ArrayList[numNN]; 
			// Create variables
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
			for (int j = 0; j < NNs[0].numInputs; j++) {
				if(i==0) {x[j][i] = model.addVar(-1, 1, lambda[j][i], GRB.CONTINUOUS, "");}
				else {x[j][i] = model.addVar(-1, 1, -lambda[j][i], GRB.CONTINUOUS, "");}
			}

			//Integrate variables
			model.update();

			// Add constraints for every NN
			// alpha_v=G_v = F_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				model.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v][i]);
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
		model.update();
		double stepsize = 0.05;

		for(int k = 0; k<50; k++ ) {
			model.optimize();
			System.out.println("LRD BOUND: "+round(model.get(GRB.DoubleAttr.ObjBound)));
			// Update Lagrangian multipliers
			double [][] diff = new double[NNs[0].numInputs][numNN];  // To capture the disagreement between variables
			for (int j = 0; j < NNs[0].numInputs; j++) {
				lambda[j][0] = 0;
				for (int i = 0; i < numNN; i++) {
					diff[j][i] = round(x[j][0].get(GRB.DoubleAttr.X)-x[j][i].get(GRB.DoubleAttr.X));
					if(!isMax) {lambda[j][i] = round(lambda[j][i]+stepsize*diff[j][i]);}
					if(isMax) {lambda[j][i] = round(lambda[j][i]-stepsize*diff[j][i]);}
					lambda[j][0] += lambda[j][i];
				}
			}

			// Update variable coeffs
			for (int i = 0; i < numNN; i++) {
				for (int j = 0; j < NNs[0].numInputs; j++) {
					if(i==0) {x[j][i].set(GRB.DoubleAttr.Obj, lambda[j][i]);}
					else {x[j][i].set(GRB.DoubleAttr.Obj, -lambda[j][i]);}
				}
			}
			model.update(); 
			stepsize = 0.05/Math.sqrt(k+1);
			/*
			for (int j = 0; j < NNs[0].numInputs; j++) {
				for (int i = 0; i < numNN; i++) {
					if(lambda[j][i] != 0) {System.out.println("LAMDBA "+j+" , "+i+": "+lambda[j][i]);}
				}
			}
			 */
		}


	}

	private void solveNode(BBNode n, GRBVar[][] x) throws GRBException {
		//Set variable bounds
		double nodeBound = 0;
		for (int i = 0; i < numNN; i++) {
			for (int j = 0; j < NNs[0].numInputs; j++) {
				x[j][i].set(GRB.DoubleAttr.LB, n.xLB[j]); 
				x[j][i].set(GRB.DoubleAttr.UB, n.xUB[j]); 
			}

		}
		model.update();
		model.optimize();
		nodeBound =  model.get(GRB.DoubleAttr.ObjBound);

		// Update node
		//System.out.println("SOLVE NODE BOUND: "+nodeBound);
		if(!isMax) {n.bound = Math.max(n.bound, round(nodeBound));}
		if(isMax) {n.bound = Math.min(n.bound, round(nodeBound));}
		//n.print();

		// Continue only if bound is better than best solution at hand
		if((isMax==false && n.bound < UB) || (isMax==true && n.bound > LB)) {
			double[] minx = new double[x.length];
			double[] maxx = new double[x.length];

			for (int j = 0; j < NNs[0].numInputs; j++) {
				minx[j] = 1;
				maxx[j] = -1;
			}

			// Record max and min x and run primal heuristic (once per node!)
			for (int i = 0; i < numNN; i++) {
				xval = new double[NNs[0].numInputs]; 
				for (int j = 0; j < NNs[0].numInputs; j++) {
					xval[j] = round(x[j][i].get(GRB.DoubleAttr.X));
					if(xval[j]<minx[j]) {minx[j] = xval[j];}
					if(xval[j]>maxx[j]) {maxx[j] = xval[j];}
				}
				if(i==0) {primalHeuristic();}
			}

			// find range of variables
			double maxDif = 0;
			double maxRange = 0;
			int branchVar = -1;
			for (int j = 0; j < NNs[0].numInputs; j++) {
				//System.out.println("min and max x: "+minx[j]+"  "+maxx[j]);
				if(maxx[j] - minx[j] >= maxDif) {
					maxDif = maxx[j] - minx[j];
					branchVar = j;
				}
				if(n.xUB[j] - n.xLB[j] > maxRange) {
					maxRange = n.xUB[j] - n.xLB[j];
				}
				
			}

			if(maxRange >= epsilon) {
					//System.out.println("BRANCH ON VARIABLE "+branchVar+" midpoint value = "+round( (maxx[branchVar] + minx[branchVar])/2.0 ));
					//Create the two new nodes
					BBNode leftn = new BBNode(n.bound, n.xLB, n.xUB, n.numBranches);
					BBNode rightn = new BBNode(n.bound, n.xLB, n.xUB, n.numBranches);
					leftn.xUB[branchVar] = round( (maxx[branchVar] + minx[branchVar])/2.0 ); 
					rightn.xLB[branchVar] = round( (maxx[branchVar] + minx[branchVar])/2.0 ); 
					tree.add(leftn);
					tree.add(rightn);
				//Update lagrangian multipliers
				double [][] diff = new double[NNs[0].numInputs][numNN];  // To capture the disagreement between variables
				double delta = stepSize/(1.0+(0.25*n.numBranches));
				//System.out.println("WEPAAAAAAAA BRANCHES "+n.numBranches+" DELTA "+delta);
				for (int j = 0; j < NNs[0].numInputs; j++) {
					lambda[j][0] = 0;
					for (int i = 0; i < numNN; i++) {
						diff[j][i] = round(x[j][0].get(GRB.DoubleAttr.X)-x[j][i].get(GRB.DoubleAttr.X));
						if(!isMax) {lambda[j][i] = round(lambda[j][i]+delta*diff[j][i]);}
						if(isMax) {lambda[j][i] = round(lambda[j][i]-delta*diff[j][i]);}
						lambda[j][0] += lambda[j][i];
					}
				}

				// Update variable coeffs
				for (int i = 0; i < numNN; i++) {
					for (int j = 0; j < NNs[0].numInputs; j++) {
						if(i==0) {x[j][i].set(GRB.DoubleAttr.Obj, lambda[j][i]);}
						else {x[j][i].set(GRB.DoubleAttr.Obj, -lambda[j][i]);}
					}
				}
				model.update(); 
				
					
				if(isMax == false) {Collections.sort(tree);}
				else {Collections.sort(tree, Collections.reverseOrder());}
			}
			else {
				//run the IP to explore the remaining space
				System.out.println("NO MORE BRANCHING VARIABLES TOO CLOSE TO EACH OTHER!!!");
				xval = new double[NNs[0].numInputs]; 
				for (int j = 0; j < NNs[0].numInputs; j++) {
					xval[j] = round( (n.xUB[j]+n.xLB[j])/2.0 );
					//System.out.println(xval[j]+" ");
					//System.out.println(minx[j]+" vs "+n.xLB[j]);
					//System.out.println(maxx[j]+" vs "+n.xUB[j]);
				}
				//System.out.println();
				primalHeuristic(); // In this case the "heuristic" is solving to optimality the remaining "space"
				numBrute ++;
			}

		}
		//else {System.out.println("PRUNNED BY BOUND!!!!!!!!!!!");}
		Depth+= n.numBranches;

	}

	private void initializePrimalHeuristic() throws GRBException {
		envS = new GRBEnv(null);
		envS.set(GRB.IntParam.OutputFlag, 0);
		modelS = new GRBModel(envS);
		if(isMax) {modelS.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

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
					int vindex = NNs[i].layers[l].get(v).ID;
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(modelS.addVar(lby, uby, 0, GRB.CONTINUOUS, ""));}
					else {y[i].add(modelS.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, ""));}
					h[i].add(modelS.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
					if(NNs[i].layers[l].get(v).fixActivation==9) {z[i].add(modelS.addVar(0, 1, 0, GRB.BINARY, "z_"+i+","+l+","+v));}
					else if(NNs[i].layers[l].get(v).fixActivation==0) {z[i].add(modelS.addVar(0, 0, 0, GRB.CONTINUOUS, "z_"+i+","+l+","+v));}
					else if(NNs[i].layers[l].get(v).fixActivation==1) {z[i].add(modelS.addVar(1, 1, 0, GRB.CONTINUOUS, "z_"+i+","+l+","+v));}
				}
			}
		}		
		for (int j = 0; j < NNs[0].numInputs; j++) {
			x[j] = modelS.addVar(-1, 1, 0, GRB.CONTINUOUS, "x_"+j);
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

		}

		// Solve
		modelS.update();

	}

	public void twoPhase(int timeL) throws GRBException {
		lambda = new double[NNs[0].numInputs][numNN];			// Lagrangian multipliers
		solFound = false;
		//Solve IP for a few minutes
		IPBigMPlus(timeL);
		timePhase1=(System.currentTimeMillis()-startTime)/1000 - timeBounding;
		//If not optimal switch to continuous branching
		double timeUsed = (System.currentTimeMillis()-startTime)/1000;
		if((UB-LB)/(Math.abs(UB)) > 0.001 && timeUsed < 3600) {
			if(solFound) {warmStartLagrangian();}
			BranchBound();
			cleanAllModels();
			timePhase2=(System.currentTimeMillis()-startTime)/1000 - timeBounding - timePhase1;
		}
		
	}
	private void cleanAllModels() throws GRBException {
		model.reset();
		model.dispose();
		modelS.reset();
		modelS.dispose();
		modelB.reset();
		modelB.dispose();
		
		env.dispose();
		envS.dispose();
		envB.dispose();
	}

	private void warmStartLagrangian() throws GRBException {
		env = new GRBEnv(null);
		env.set(GRB.IntParam.OutputFlag, 0);
		model = new GRBModel(env);
		//model.set(GRB.DoubleParam.TimeLimit, 60.0);
		if(isMax) {model.set(GRB.IntAttr.ModelSense, GRB.MAXIMIZE);}

		GRBVar [][] x = new GRBVar[NNs[0].numInputs][numNN];
		for (int i = 0; i < numNN; i++) {

			ArrayList<GRBVar> [] y = new ArrayList[numNN]; 
			ArrayList<GRBVar> [] h = new ArrayList[numNN]; 
			// Create variables
			y[i] = new ArrayList<GRBVar>();
			h[i] = new ArrayList<GRBVar>();
			for (int l = 0; l < NNs[i].numLayers; l++) {
				for (int v = 0; v < NNs[i].layers[l].size(); v++) {
					double lby =  NNs[i].layers[l].get(v).lb;
					double uby = Math.max(0, NNs[i].layers[l].get(v).ub); // y variables are nonnegative!!!
					if(l < NNs[i].numLayers-1) {y[i].add(model.addVar(lby, uby, 0, GRB.CONTINUOUS, "y_"+i+","+l+","+v));}
					else {y[i].add(model.addVar(lby, uby, (1/(numNN+0.0)), GRB.CONTINUOUS, "y_"+i+","+l+","+v));}
					h[i].add(model.addVar(NNs[i].layers[l].get(v).lb, NNs[i].layers[l].get(v).ub, 0, GRB.CONTINUOUS, ""));
				}
			}
			for (int j = 0; j < NNs[0].numInputs; j++) {
				if(i==0) {x[j][i] = model.addVar(-1, 1, lambda[j][i], GRB.CONTINUOUS, "");}
				else {x[j][i] = model.addVar(-1, 1, -lambda[j][i], GRB.CONTINUOUS, "");}
			}

			//Integrate variables
			model.update();

			// Add constraints for every NN
			// alpha_v=G_v = F_v for the first layer
			for (int v = 0; v < NNs[i].layers[0].size(); v++) {
				GRBLinExpr expr = new GRBLinExpr();
				int index = NNs[i].layers[0].get(v).ID;
				expr.addTerm(1, y[i].get(index));
				expr.addTerm(-1, h[i].get(index));
				model.addConstr(expr, GRB.EQUAL, 0, "");
				GRBLinExpr expr2 = new GRBLinExpr();
				expr2.addTerm(1, h[i].get(index));
				expr2.addTerm(-1, x[v][i]);
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
					double lb = -NNs[i].layers[l].get(v).lb;
					double ub = NNs[i].layers[l].get(v).ub;

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
					model.addConstr(expr4, GRB.LESS_EQUAL, lb*(1-zval[i].get(vindex)), "");

					GRBLinExpr expr = new GRBLinExpr();
					expr.addTerm(1, y[i].get(vindex));
					model.addConstr(expr, GRB.LESS_EQUAL, ub*zval[i].get(vindex), "");

				}
			}
		}
		model.update();
				
		double stepsize = 0.1;	
		for(int k = 0; k<20; k++ ) {
			model.optimize();
			System.out.println("LRD BOUND: "+round(model.get(GRB.DoubleAttr.ObjBound)));
			
			// Update Lagrangian multipliers
			double [][] diff = new double[NNs[0].numInputs][numNN];  // To capture the disagreement between variables
			for (int j = 0; j < NNs[0].numInputs; j++) {
				lambda[j][0] = 0;
				for (int i = 0; i < numNN; i++) {
					diff[j][i] = round(x[j][0].get(GRB.DoubleAttr.X)-x[j][i].get(GRB.DoubleAttr.X));
					if(!isMax) {lambda[j][i] = round(lambda[j][i]+stepsize*diff[j][i]);}
					if(isMax) {lambda[j][i] = round(lambda[j][i]-stepsize*diff[j][i]);}
					lambda[j][0] += lambda[j][i];
				}
			}

			// Update variable coeffs
			for (int i = 0; i < numNN; i++) {
				for (int j = 0; j < NNs[0].numInputs; j++) {
					if(i==0) {x[j][i].set(GRB.DoubleAttr.Obj, lambda[j][i]);}
					else {x[j][i].set(GRB.DoubleAttr.Obj, -lambda[j][i]);}
				}
			}
			model.update(); 
			stepsize = 0.1/(1+k);
			/*
			for (int j = 0; j < NNs[0].numInputs; j++) {
				for (int i = 0; i < numNN; i++) {
					if(lambda[j][i] != 0) {System.out.println("LAMDBA "+j+" , "+i+": "+lambda[j][i]);}
				}
			}
			 */
		}	 
		
		timeGradient = (System.currentTimeMillis()-startTime)/1000 - (timeBounding+timePhase1);
		System.out.println("TIME DESCENT: "+timeGradient+" bound 2: "+bound2);
		
		model.reset();
		model.dispose();
		env.dispose();
	}

	public void IPBigMPlus(int timeL) throws GRBException {
		env.set(GRB.IntParam.OutputFlag, 0);
		env.set(GRB.IntParam.LazyConstraints, 1);
		model = new GRBModel(env);
		// Turn off all prepro for sensitivity analysis only!
		//model.set(GRB.IntParam.Presolve, 0);
		//model.set(GRB.IntParam.Cuts, 0);
				
		double timeUsed = (System.currentTimeMillis()-startTime)/1000;
		double timeLeft = Math.min(timeL, 3600-timeUsed);
		model.set(GRB.DoubleParam.TimeLimit, timeLeft);
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
			if(!isMax) {
				UB = round(model.get(GRB.DoubleAttr.ObjVal));
				LB = round(model.get(GRB.DoubleAttr.ObjBound));
				bound1 = LB;
				
			}
			else {
				LB = round(model.get(GRB.DoubleAttr.ObjVal));
				UB = round(model.get(GRB.DoubleAttr.ObjBound));
				bound1 = UB;
				
			}
			System.out.println("PHASE ONE DONE! "+" LB = "+LB+" UB = "+UB+" GAP: "+round( (UB-LB)/(Math.abs(UB))) + "BOUND 1: "+bound1);

			// Record Z variables
			if(model.get(GRB.IntAttr.SolCount) >= 1) {
				solFound = true;
				for (int i = 0; i < numNN; i++) {
					zval[i].clear();
					for (int j = 0; j < z[i].size(); j++) {
						if(z[i].get(j).get(GRB.DoubleAttr.X) > 0.9) {
							zval[i].add(1);
						}
						else {
							zval[i].add(0);
						}
					}	
				}
			}
			// ONLY UNCOMMENT FOR SENSITIVITY ANALISIS
			//opt = model.get(GRB.DoubleAttr.ObjVal);
			//gap = model.get(GRB.DoubleAttr.MIPGap);
			
			model.reset();
			model.dispose();
			env.dispose();
			envS.dispose();
			/*for (int i = 0; i < NNs[0].numInputs; i++) {
				System.out.print(x[i].get(GRB.DoubleAttr.X)+" ");
			}
			System.out.println();
			 */
		}
	}





}
