package twoPhaseSensitivity;
import java.util.ArrayList;

public class bendersCut {

	double constant; 			// Bias
	String type;        		// Opt or Feasibility
	ArrayList<Double>[] coef0;	// Coefficients for the z variables == 0
	ArrayList<Double>[] coef1;	// Coefficients for the z variables == 1

	
	public bendersCut(int n, double c, String t) {
		constant = c;
		type = t;
		coef0 = new ArrayList[n];
		coef1 = new ArrayList[n];
		for (int i = 0; i < n; i++) {
			coef0[i] = new ArrayList<Double>();
			coef1[i] = new ArrayList<Double>();
		}
	}

	public void print() {
		System.out.println("Constant: "+constant);
		for (int i = 0; i < coef0.length; i++) {
			for (int j = 0; j < coef0[i].size(); j++) {
				System.out.print(coef0[i].get(j)+" ");
			}
			System.out.println();
		}
		for (int i = 0; i < coef1.length; i++) {
			for (int j = 0; j < coef1[i].size(); j++) {
				System.out.print(coef1[i].get(j)+" ");
			}
			System.out.println();
		}

		
	}


}
