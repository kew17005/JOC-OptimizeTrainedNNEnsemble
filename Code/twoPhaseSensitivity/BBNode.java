package twoPhaseSensitivity;

public class BBNode implements Comparable<BBNode> {

	double[] xLB;
	double[] xUB;	
	double bound; 		
	int numBranches;
	
	
	public BBNode(double _bound, double[] _xlb, double[] _xub, int numB) {
		bound = _bound;
		xLB = new double[_xlb.length];
		xUB = new double[_xub.length];
		for (int j = 0; j < _xub.length; j++) {
			xLB[j] = _xlb[j];
			xUB[j] = _xub[j];
		}
		numBranches = numB+1;
	}

	public void print() {
		System.out.println("BB NODE BOUND: "+bound);
		for (int i = 0; i < xLB.length; i++) {
			System.out.print(" x_"+i+" ["+xLB[i]+" , "+xUB[i]+"] ");
		}
		System.out.println();
	}
	
	@Override
	public int compareTo(BBNode n) {
		if(this.bound > n.bound) {return 1;}
		else if(this.bound < n.bound) {return -1;}
		else{return 0;}
	}

}
