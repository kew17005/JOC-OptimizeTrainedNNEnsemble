package twoPhaseAlg;
import java.io.IOException;

import gurobi.GRB;
import gurobi.GRBException;


public class main {

	public static void main(String[] args) throws NumberFormatException, IOException, GRBException {

		java.io.PrintStream ps = new java.io.PrintStream( new java.io.FileOutputStream("ResultsTwoPhase2022.txt", true));

		int[] nodes = {20, 40};

		//Iterate over the network sizes
		for (int q = 3; q <= 5; q+=2) {
			for (int l = 2; l <= 4 ; l+=2) {
				for (int i = 0; i <= 1 ; i++) {
					//Iterate over the random instances	
					for	(int s = 1; s <= 3 ; s++) {
						// Input the x variables range
						algHandler alg = new algHandler(-1,1);
						alg.readInstance("data/Beale/Beale_"+q+"_"+l+"_"+nodes[i]+"_"+s+".txt");
						alg.startTime = System.currentTimeMillis();
						//alg.isMax = true;							// true if maximization problem
						
						alg.boundLP();
						alg.survey();
						alg.boundFestTargeted();
						alg.timeBounding = (System.currentTimeMillis()-alg.startTime)/1000;
						alg.twoPhase(180);
						System.out.println("TwoPhase "+"Time "+(System.currentTimeMillis()-alg.startTime)/1000);
						ps.println("TwoPhaseNEW Beale"+alg.numNN+" "+l+" "+nodes[i]+" "+s+" "+(System.currentTimeMillis()-alg.startTime)/1000+" "+alg.opt+" "+alg.gap+" "+alg.timeBounding+" "+alg.timePhase1+" "+alg.timePhase2+" "+alg.numberNodesExplored);	
						//ps.print("TwoPhaseNEW Peaks "+alg.numNN+" "+l+" "+nodes[i]+" "+s+" "+(System.currentTimeMillis()-alg.startTime)/1000+" "+alg.opt+" "+alg.gap+" SOLUTION: ");
						//for (int j = 0; j < alg.NNs[0].numInputs; j++) {
						//	ps.print(alg.xstar[j]+" ") ;
						//}
						//ps.println();
						System.gc();
						
					}
				}
			}
		}
	}
}
