package branchCutAlg;
import java.io.IOException;

import gurobi.GRBException;

public class main {

	public static void main(String[] args) throws NumberFormatException, IOException, GRBException {

		java.io.PrintStream ps = new java.io.PrintStream( new java.io.FileOutputStream("BCv2WEPA.txt", true));

		int[] nodes = {20, 40};

		//Iterate over the network sizes
		for (int q = 3; q <= 5 ; q+=2) {
			for (int l = 2; l <= 4 ; l+=2) {
				for (int i = 0; i <= 1 ; i++) {
					//Iterate over the random instances	
					for	(int s = 1; s <= 3 ; s++) {
						algHandler alg = new algHandler();
						alg.readInstance("data/Beale/Beale_"+q+"_"+l+"_"+nodes[i]+"_"+s+".txt");
						alg.startTime = System.currentTimeMillis();
						//alg.isMax = true; 							// true if maximization problem
						//alg.boundIP(-1, 1);  //USE ALONE NOT WITH boundLP or Survey

						alg.boundLP(-1, 1);
						//alg.survey();
						//alg.boundTargeted(-1, 1);
						alg.getCoeff();
						alg.timeBounding = (System.currentTimeMillis()-alg.startTime)/1000;
						alg.IPBC();
						alg.timeIP = (System.currentTimeMillis()-alg.startTime)/1000 - alg.timeBounding;
						System.out.println("B&C "+"Time "+(System.currentTimeMillis()-alg.startTime)/1000);
						//ps.println("B&C Beale "+alg.numNN+" "+l+" "+nodes[i]+" "+s+" "+(System.currentTimeMillis()-alg.startTime)/1000+" "+alg.opt+" "+alg.gap+" "+" "+alg.rootBound+" "+alg.afterCutsBound+" "+alg.numCritical);
						ps.println("B&C Beale"+alg.numNN+" "+l+" "+nodes[i]+" "+s+" "+(System.currentTimeMillis()-alg.startTime)/1000+" "+alg.opt+" "+alg.gap+" "+alg.timeBounding);
						//alg.cleanModels();
						System.gc();
					}
				}
			}
		}
	}
}
