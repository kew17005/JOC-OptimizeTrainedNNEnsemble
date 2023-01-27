package twoPhaseSensitivity;
import java.io.IOException;
import gurobi.GRBException;


public class main {

	public static void main(String[] args) throws NumberFormatException, IOException, GRBException {

		java.io.PrintStream ps = new java.io.PrintStream( new java.io.FileOutputStream("TwoPhaseSEN.txt", true));

		int[] nodes = {40};

		//Iterate over the network sizes
		for (int q = 5; q <= 5 ; q+=2) {
			for (int l = 4; l <= 4 ; l+=2) {
				for (int i = 0; i <= 0 ; i++) {
					//Iterate over the random instances	
					for	(int s = 1; s <= 3 ; s++) {
						algHandler alg = new algHandler();
						alg.readInstance("data/Peaks/Peaks_"+q+"_"+l+"_"+nodes[i]+"_"+s+".txt");
						alg.startTime = System.currentTimeMillis();
						//alg.isMax = true;							// true if maximization problem
						//alg.boundFest(-1, 1);  USE ALONE NOT WITH boundLP and Survey

						alg.boundLP(-1, 1);
						alg.survey();
						alg.boundFestTargeted(-1, 1);
						alg.timeBounding = (System.currentTimeMillis()-alg.startTime)/1000;
						alg.twoPhase(180);
						System.out.println("TwoPhase "+"Time "+(System.currentTimeMillis()-alg.startTime)/1000);
						ps.println("Peaks "+alg.numNN+"_"+l+"_"+nodes[i]+"_"+s+" "+(System.currentTimeMillis()-alg.startTime)/1000+" "+alg.opt+" "+alg.bound1+" "+alg.bound2+" "+alg.timeGradient+" "+alg.numBrute+" "+(alg.Depth/(alg.numberNodesExplored+0.0)));
						//ps.println("YESVIS Perm "+alg.numNN+" "+l+" "+nodes[i]+" "+s+" "+(System.currentTimeMillis()-alg.startTime)/1000+" "+alg.opt+" "+alg.gap+" "+alg.timeBounding+" "+alg.timePhase1+" "+alg.totalCuts);	
						
						System.gc();
						
						// How many nodes we solve via brute force?
						// How deep do we go?
						// 
						
					}
				}
			}
		}
	}
}
