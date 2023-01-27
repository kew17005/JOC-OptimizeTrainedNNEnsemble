package twoPhaseSensitivity;

public class Node {

	double[] weights;	// Weights with respect to all the nodes in the previous layer
	double bias; 		// Bias
	int ID;
	int dim;			// Number of inputs
	int layer;			// Layer containing the node
	
	double lb;			// Lower bound on the pre activation function
	double ub;			// Upper bound on the pre activation function

	double lbLocal;			// Lower bound on the pre activation function for fixed z
	double ubLocal;			// Upper bound on the pre activation function for fixed Z

	int fixActivation;  // 0 if always inactive, 1 if always active, 9 is not fixed
	
	double L [];		// Lhat constants from MP paper
	double U [];		// Uhat constants from MP paper
	
	double numCritical;
	double violations;

	
	public Node(int n, int id, int l) {
		ID = id;
		dim = n;
		layer = l;
		weights =  new double[n];
		L =  new double[n];
		U =  new double[n];
		bias = 0;
		lb = -1000;
		ub = 1000;
		lbLocal = -1000;
		ubLocal = 1000;

		numCritical = 0;
		violations = 0;
		fixActivation = 9;
	}

	public void print() {
		System.out.print("NODE ID: "+ID+" Weights: ");
		for (int i = 0; i < weights.length; i++) {
			System.out.print(weights[i]+" ");
		}
		System.out.print("Bias: "+bias);
		
	}
	public void printBounds() {
		System.out.println("NODE ID: "+ID+" ["+lb+" , "+ub+"]"+" fixed: "+fixActivation+" Critical: "+numCritical+" AvgViol: "+(violations/numCritical));
	}

	public void printBoundsL() {
		// TODO Auto-generated method stub
		System.out.println("NODE ID: "+ID+" ["+lbLocal+" , "+ubLocal+"]");
	}


}
