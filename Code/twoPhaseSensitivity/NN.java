package twoPhaseSensitivity;
import java.util.ArrayList;

public class NN {

	int numLayers;				// Number of layers
	int numNodes;				// Total number of nodes
	int numInputs;				// Number of features
	int[] config;				// Number of nodes in each layer
	ArrayList<Node>[] layers;	// Nodes in each layer
	
	public NN(int n) {
		numLayers = n;
		config = new int[numLayers];
		layers = new ArrayList[numLayers];
		for (int i = 0; i < numLayers; i++) {
			layers[i] = new ArrayList<Node>();
		}
	}
	
	public void print() {
		System.out.println("Number of Layers: "+numLayers +"********************************************************");
		System.out.print("Configuration: " );
		for (int i = 0; i < config.length; i++) {
			System.out.print(config[i]+" ");
		}
		System.out.println();
		for (int i = 0; i < numLayers; i++) {
			System.out.println("Layer "+i);
			for (int j = 0; j < config[i]; j++) {
				System.out.print("node "+j+": ");
				layers[i].get(j).print();
				System.out.println();
			}
			
		}
		
	}

	public void printBounds() {
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < config[i]; j++) {
				layers[i].get(j).printBounds();
			}
			
		}
		
	}

	public void computeCoeff() {
		//Compute for all layers except the input layer and the output layer
		// first layer is different because y can be negative
		for (int l = 1; l <= 1; l++) {
			for (int v = 0; v < layers[l].size(); v++) {
				Node n = layers[l].get(v);
				for (int i = 0; i < n.dim; i++) {
					if(n.weights[i] >= 0) {n.L[i] =  layers[l-1].get(i).lb;} 
					else {n.L[i] = layers[l-1].get(i).ub;}

					if(n.weights[i] >= 0) {n.U[i] = layers[l-1].get(i).ub;}
					else {n.U[i] = layers[l-1].get(i).lb;}
					//System.out.println("Node "+n.ID+" L_"+i+": "+n.L[i]+" U_"+i+": "+n.U[i]);
				}
			}
		}
		for (int l = 2; l < numLayers-1; l++) {
			for (int v = 0; v < layers[l].size(); v++) {
				Node n = layers[l].get(v);
				for (int i = 0; i < n.dim; i++) {
					if(n.weights[i] >= 0) {n.L[i] =  Math.max(0,layers[l-1].get(i).lb);} // Nonnegative after relu!
					else {n.L[i] =  Math.max(0,layers[l-1].get(i).ub);}

					if(n.weights[i] >= 0) {n.U[i] =  Math.max(0,layers[l-1].get(i).ub);}
					else {n.U[i] =  Math.max(0,layers[l-1].get(i).lb);}
					//System.out.println("Node "+n.ID+" L_"+i+": "+n.L[i]+" U_"+i+": "+n.U[i]);
				}
			}
		}

		
	}

	public void printBoundsL() {
		// TODO Auto-generated method stub
		for (int i = 0; i < numLayers; i++) {
			for (int j = 0; j < config[i]; j++) {
				layers[i].get(j).printBoundsL();
			}
			
		}
	}
	
}
