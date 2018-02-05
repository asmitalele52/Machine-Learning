import  java.util.*;
import  java.io.*;

public class Node {
	public Node leftNode;
    public Node rightNode;
    public int depth;
    public ArrayList<ArrayList<Integer>> dataSet;
    public int noOfPositives;
    public int noOfnegatives;
    public String name;
    public int label;
    public double entropy;
    public int nodeNumber; 
    public static int totalNodes;
    public ArrayList<Integer> potentialSplits;
    
    //initializing the values the default values and incrementing node count
    public Node(){
    	dataSet = new ArrayList<ArrayList<Integer>>();
    	potentialSplits = new ArrayList<Integer>();
    	nodeNumber = Node.totalNodes;
    	Node.totalNodes++;
    	label = -1;
    }
    
    //calculate and return entropy based on the number of positive and negative labels
    public static double CalculateEntropy(int positives, int negatives) {
    	double entropy = 1.0;
    	double pb1 = positives !=0 ? (double) positives/(positives+negatives) : 0.0;
    	double pb2 = negatives !=0 ? (double) negatives/(positives+negatives): 0.0;
    	
    	double logpb1 = pb1 != 0.0 ? (Math.log10(pb1)/Math.log10(2)) : 0.0;
    	double logpb2 = pb2 != 0.0 ? (Math.log10(pb2)/Math.log10(2)) : 0.0;
    	
    	entropy = ( (pb1 * logpb1) + (pb2 * logpb2) ) * -1; 
    	
    	return entropy;
    }

   //set the properties of the created node - data, name, depth, entropy and potential attributes that the given node can split on
   public static Node constructNode(Node node, ArrayList<ArrayList<Integer>> data, ArrayList<Integer> potentialAttributes, int depth, String name) {
    	node.dataSet = data;
    	node.name = name;
    	node.depth = depth;
    	int positives = 0, negatives = 0;
    	if(data != null) {
    	int attributeLength = data.get(0).size();
    	
    	for(ArrayList<Integer> entry : data) {
    		if(entry.get(attributeLength-1) == 1) {
    			positives++;
    		}
    		else {
    			negatives++;
    		}
    	  }
    	
    	node.entropy = CalculateEntropy(positives,negatives);
    	node.noOfPositives = positives;
    	node.noOfnegatives = negatives;
    	}
    	else {
    	node.entropy = 0.0;  	
    	node.noOfPositives = 0;
    	node.noOfnegatives = 0;
    	}
    	node.potentialSplits = potentialAttributes;
    	return node;
    }
    
   	//calculate information gain based on the split attribute chosen
    public static double CalculateInformationGain(ArrayList<ArrayList<Integer>> data, double entropy, int attributeNumber) {
    	double informationGain= 0.0;
    	int noOfPostives0 = 0, noOfNegatives0 = 0, noOfPositives1 = 0, noOfNegatives1 = 0;
    	int attributeLength = data.get(0).size();
    	for(ArrayList<Integer> entry : data) {
    		if(entry.get(attributeNumber) == 0) {
    			if(entry.get(attributeLength-1) == 0) {
    				noOfNegatives0++;
    			}
    			else {
    				noOfPostives0++;
    			}
    		}
    		else {
    			if(entry.get(attributeLength-1) == 0) {
    				noOfNegatives1++;
    			}
    			else {
    				noOfPositives1++;
    			}
    		}
    	}
    	double leftEntropy = CalculateEntropy(noOfPostives0,noOfNegatives0);
    	double rightEntropy = CalculateEntropy(noOfPositives1, noOfNegatives1);
    	double weightedAvg = ((( (double) (noOfPostives0 + noOfNegatives0))/ (double) data.size())*leftEntropy) + 
    			                 ((((double) (noOfPositives1+noOfNegatives1))/(double) data.size())*rightEntropy);
    	informationGain = entropy - weightedAvg;		
    	return informationGain;
    }
    
    //choosing the best attribute to split a node which is not pure. Choose the node with the highest information gain
    public static int BestAttributeToSplit(Node node) {
    	int attributeNumber = 0;
    	double informationGain = -1 * Double.MAX_VALUE;
    	double tempInformationGain = 0.0;
    	for(int attribute : node.potentialSplits) {
    		tempInformationGain = CalculateInformationGain(node.dataSet, node.entropy, attribute);
    		if(tempInformationGain >= informationGain) {
    			informationGain = tempInformationGain;
    			attributeNumber = attribute;
    		}
    	}
    	return attributeNumber;
    }
    
    //return the data considering the input attribute as 0/1
    public static ArrayList<ArrayList<ArrayList<Integer>>> SplitData(ArrayList<ArrayList<Integer>> data, int attribute){
    	ArrayList<ArrayList<ArrayList<Integer>>> splitData = new ArrayList<ArrayList<ArrayList<Integer>>>();
    	ArrayList<ArrayList<Integer>> positiveData = new ArrayList<ArrayList<Integer>>();
    	ArrayList<ArrayList<Integer>> negativeData = new ArrayList<ArrayList<Integer>>();
    	for(ArrayList<Integer> entry : data) {
    		if(entry.get(attribute) == 0) {
    			negativeData.add(entry);
    		}
    		else {
    			positiveData.add(entry);
    		}  			
    	}
    	splitData.add(negativeData);
    	splitData.add(positiveData);
    	return splitData; 
    }
    
    /* recursively create decision true
     * base case : if a node is pure - label the node based on the purity
     * Recursive step: get the best attribute to split for the node, recursively create the tree for left and right subtree after the data is split on the select attribute
     * extra case: if the available atteibutes to split the tree are over but the data is still not pure, then count the maximum no of nodes and give label of the majority class
     */
    public static Node createDecisionTree(Node node, HashMap<Integer,String> attributeNames) {
    	if(node.noOfPositives == 0 || node.noOfnegatives ==0) {
    		node.label = (node.noOfPositives > 0) ? 1 : 0;
        	return node;
    	}
    	else {
    		if(node.potentialSplits.size() > 0) {
    		int splitAttribute = BestAttributeToSplit(node);
    		Node leftNode = new Node();
    		ArrayList<Integer> potentialAttr = new ArrayList<Integer>(node.potentialSplits);
    		potentialAttr.remove((Integer)splitAttribute);
    		ArrayList<ArrayList<ArrayList<Integer>>> dataSets = SplitData(node.dataSet, splitAttribute);
    		if(dataSets.get(0).size() > 0) {
    		leftNode = constructNode(leftNode,dataSets.get(0), potentialAttr, (node.depth+1), attributeNames.get(splitAttribute));
    		leftNode = createDecisionTree(leftNode, attributeNames);
    		}
    		else {
    			leftNode = constructNode(leftNode,null, potentialAttr, (node.depth+1), attributeNames.get(splitAttribute));
    			leftNode.label = 0;
    	    }
    		Node rightNode = new Node();
    		if(dataSets.get(1).size() > 0) {
    		rightNode = constructNode(rightNode,dataSets.get(1), potentialAttr, (node.depth+1), attributeNames.get(splitAttribute));
    		rightNode = createDecisionTree(rightNode, attributeNames);
    		}
    		else {
    			rightNode = constructNode(rightNode, null , potentialAttr, (node.depth+1), attributeNames.get(splitAttribute));
    			rightNode.label = 0;
    		}    		    		    		
    		node.leftNode = leftNode;
    		node.rightNode = rightNode;
    		return node;
    		}
    		else {
    			if(node.noOfPositives >= node.noOfnegatives) {
    				node.label = 1;
    			}
    			else {
    				node.label = 0;
    			}
    			return node;
    		}
    	}  	
    }
    
    // function that will print the decision tree
    public static void printDecisionTree(Node node) {
    	if(node != null) {
    		  if(node.label != -1) {
    			System.out.print(" " + node.label);
    			return;
    		  }
    		  else {
    			  int depth = node.leftNode.depth;
    			  String s = "";
    			  while(depth > 1) {
    				  s += "| ";
    				  depth--;
    			  }
    			  System.out.print("\n"+ s + node.leftNode.name + " = 0 :");
    		  }
    		printDecisionTree(node.leftNode);
      		  if(node.label != -1) {
      			System.out.print(" " + node.label);
      			return;
      		  }
      		  else {
      			int depth = node.rightNode.depth;
  			    String s = "";
  			    while(depth > 1) {
  				  s+= "| ";
  				  depth--;
  			    }
      			  System.out.print("\n" + s + node.rightNode.name + " = 1 :");
      		  }
    		printDecisionTree(node.rightNode);
    	}
    }
    
    //check the label for any given set of input and if the label of the given input matches with the tree nodes
    public static boolean CheckLabel(Node node, ArrayList<Integer> entry, HashMap<String, Integer> attributeNumbers)
    {
    	int rowSize = entry.size();
		if(node.label != -1) {
			if(entry.get(rowSize-1) == node.label)
				return true;
			else
				return false;
		}
		else
		{
			int splitAttribute = attributeNumbers.get(node.leftNode.name);
			if(entry.get(splitAttribute) == 0)
				return CheckLabel(node.leftNode, entry, attributeNumbers);
			else
				return CheckLabel(node.rightNode, entry, attributeNumbers);
		}
    }
    
    //calculate accuracy of the given data with the designed tree
    public static double GetDataAccuracy(Node node, ArrayList<ArrayList<Integer>> dataToCheck, HashMap<String, Integer> attributeNumbers)
    {
    	int count = 0;
    	if(dataToCheck.size() <= 0)
    		return 0.0;
    	int sizeOfDataRow = dataToCheck.get(0).size();
    	for (ArrayList<Integer> entry : dataToCheck)
    	{
    		if(node.label != -1)
    		{
    			if(entry.get(sizeOfDataRow-1) == node.label)
    				count++;
    		}
    		else
    		{
    			int splitAttribute = attributeNumbers.get(node.leftNode.name);
    			if(entry.get(splitAttribute) == 0) {
    				if(CheckLabel(node.leftNode, entry, attributeNumbers)) {
    					count++;
    				}
    			}
    			else {
    				if(CheckLabel(node.rightNode, entry, attributeNumbers)) {
    					count++;
    				}
    		    }
    	   }
    	}
    	return ((double)count)/(double)(dataToCheck.size());
    }
    
    //populate data for the tree from the path provided
    public static ArrayList<ArrayList<Integer>> GetData(String dataPath) throws IOException{
    	ArrayList<ArrayList<Integer>>  data = new ArrayList<ArrayList<Integer>>();
    	ArrayList<Integer> temp;
    	BufferedReader input = null;
    	try {
    		input = new BufferedReader(new InputStreamReader(new FileInputStream(dataPath), "UTF-8"));
    		}catch(Exception e) {
    			System.out.println(e);
    		}
    		String currentLine = "";
    		input.mark(1);
    		if (input.read() != 0xFEFF)
    		  input.reset();
    		currentLine = (input.readLine());		
    		while((currentLine = input.readLine())!= null) {
    			String [] entries = currentLine.split(",");
    			 temp = new ArrayList<Integer>();
    			for(String item : entries) {
    				temp.add(Integer.parseInt(item));
    			}
    			data.add(temp);
    		}
    	return data;
    }
    
    //prune the node fromt the subtree
    public static void PruneNode(Node node, int nodeNumberToPrune) {
    	if(node != null) {
    		if(node.nodeNumber == nodeNumberToPrune) {
    			if(node.label == -1) {
    				node.label = (node.noOfPositives >= node.noOfnegatives) ? 1 : 0 ;
    				node.leftNode = null;
    				node.rightNode = null;
    				return;
    			}
    		}
    		PruneNode(node.leftNode,nodeNumberToPrune);
    		PruneNode(node.rightNode,nodeNumberToPrune);
    	}
    }
    
    //prune the tree with the node numbers that provided in the list
    public static Node PruneDecisionTree(Node root, ArrayList<Integer> nodesToPrune, int totalNodes) {
    	int nodeNumberToPrune = 0;
          while(nodesToPrune.size() > 0) {
    		nodeNumberToPrune = nodesToPrune.get(0);
    	    PruneNode(root, nodeNumberToPrune);
    	    nodesToPrune.remove(0);
    	}
    	return root;
    }
    
    //get pruning nodes based on the pruning factor, function will return randomly selected node numbers to be pruned
    public static ArrayList<Integer> GetPruningNodes(int totalNodes, double pruningFactor)
    {
    	Set<Integer> pruningNodesSet = new LinkedHashSet<Integer>();
    	ArrayList<Integer> pruningNodesList = new ArrayList<>(); 
    	int noOfNodesToPrune = (int) (pruningFactor*totalNodes);
    	Random r = new Random();
    	while(pruningNodesSet.size() < noOfNodesToPrune) {
    		pruningNodesSet.add(r.nextInt(totalNodes-1) + 1);
    		
    	}
    	pruningNodesList.addAll(pruningNodesSet);
    	Collections.sort(pruningNodesList);
    	//System.out.println(pruningNodesList);
    	return pruningNodesList;
    }
    
    //get the count of total nodes in a tree
    public static int GetTotalNodes(Node node) {
    	if(node != null) {
    		return 1 + GetTotalNodes(node.leftNode) + GetTotalNodes(node.rightNode);
    	}
    	return 0;
    }
    
  //get the count of leaf nodes in a tree
    public static int GetTotalLeafNodes(Node node) {
    	if(node != null) {
    		if(node.label != -1) {
    			return 1;
    		}
    		else
    		{
    			return GetTotalLeafNodes(node.leftNode) + GetTotalLeafNodes(node.rightNode);
    		}
    	}
    	return 0;
    }
    
  //get the count of total nodes and leaf nodes in a tree
    public static ArrayList<Integer> GetNoOfNodesInTree(Node root){
    	 int totalNodes = 1 + GetTotalNodes(root.leftNode) + GetTotalNodes(root.rightNode);
    	 int totalLeafNodes =  GetTotalLeafNodes(root.leftNode) + GetTotalLeafNodes(root.rightNode);
    	 ArrayList<Integer> totalNodeList = new ArrayList<Integer>();
    	 totalNodeList.add(totalNodes);
    	 totalNodeList.add(totalLeafNodes);
    	 return totalNodeList;
    }
    
    /*
     * Program starts here
     * ID3 algorithm implementated
     * constructed the tree with training set, and tested using test and validation data
     * the tree is pruned and accuracies are found again
     */
	public static void main(String[] args) throws IOException {		
		Node root = new Node();
		BufferedReader input = null;
		ArrayList<ArrayList<Integer>> trainingData;
		ArrayList<ArrayList<Integer>> validationData;
		ArrayList<ArrayList<Integer>> testData;
		ArrayList<Integer> nodesToPrune;
		HashMap<Integer,String> attributeNames = new HashMap<Integer,String>();
		HashMap<String, Integer> attributeNumbers = new HashMap<String,Integer>();
		String trainingDataPath = args[0], validationDataPath = args[1], testDataPath = args[2];
		double pruningFactor = Double.parseDouble(args[3]);

		//Data pre processing
		try {
		input = new BufferedReader(new InputStreamReader(new FileInputStream(trainingDataPath), "UTF-8"));
		}catch(Exception e) {
			System.out.println(e);
		}
		String currentLine = "";
		input.mark(1);
		if (input.read() != 0xFEFF)
		  input.reset();
		currentLine = (input.readLine());
		String[] attrNames = currentLine.split(",");
		ArrayList<Integer> potentialAttributes=  new ArrayList<Integer>();
		for(int i=0; i< attrNames.length;i++) {
			attributeNames.put(i, attrNames[i]);
			attributeNumbers.put(attrNames[i], i);
			potentialAttributes.add(i);
		}
		potentialAttributes.remove(potentialAttributes.size() - 1 );		
		trainingData = GetData(trainingDataPath);
		//printData(trainingData);
		validationData = GetData(validationDataPath);
		testData = GetData(testDataPath);
		//Data preprocessing Done
        
        root = constructNode(root, trainingData, potentialAttributes, 0, "root");
        root = createDecisionTree(root, attributeNames);
        System.out.println("\n Decision Tree Before Pruning \n");
        printDecisionTree(root);
        ArrayList<Integer> totalNodesList = GetNoOfNodesInTree(root);
        
        int nodesInTree = totalNodesList.get(0);
        int leafNodesInTree = totalNodesList.get(1);
        
        //Get accuracy of decision tree model constructed above on training,validation,test data.
        double trainingAccuracy = GetDataAccuracy(root, trainingData, attributeNumbers); // traingDataAccuracy
        double validationAccuracy = GetDataAccuracy(root, validationData,attributeNumbers); // traingDataAccuracy
        double testAccuracy = GetDataAccuracy(root, testData, attributeNumbers); // traingDataAccuracy
        
        System.out.println("\n");
        System.out.println("Pre-Pruned Accuracy\n--------------------------------");
        System.out.println("Number of training instances = " + trainingData.size());
        System.out.println("Number of training attributes = " + (trainingData.get(0).size()-1));
        System.out.println("Total number of nodes in the tree = " + nodesInTree);
        System.out.println("Number of leaf nodes in the tree = " + leafNodesInTree);
        System.out.println("Accuracy of the model on traianing dataset = " + trainingAccuracy);
        System.out.println("\nNumber of validation instances = " + (validationData.size()));
        System.out.println("Number of validation attributes = " + (validationData.get(0).size()-1));
        System.out.println("Accuracy of the model on validation dataset before pruning = " + validationAccuracy);
        System.out.println("\nNumber of testing instances = " + (testData.size()));
        System.out.println("Number of testing attributes = " + (testData.get(0).size()-1));
        System.out.println("Accuracy of the model on testing dataset = " + testAccuracy);
        
        
        nodesToPrune = GetPruningNodes(Node.totalNodes,pruningFactor);
        root = PruneDecisionTree(root, nodesToPrune, Node.totalNodes);
        
        System.out.println("\n Decision Tree After Pruning \n");
        printDecisionTree(root);
        
        
        ArrayList<Integer> totalNodesListAfterPruning = GetNoOfNodesInTree(root);
        
        int nodesInTreeAfterPruning = totalNodesListAfterPruning.get(0);
        int leafNodesInTreeAfterPruning = totalNodesListAfterPruning.get(1);
        
        double trainingAccuracyAfterPruning = GetDataAccuracy(root, trainingData, attributeNumbers); // traingDataAccuracy
        double validationAccuracyAfterPruning = GetDataAccuracy(root, validationData,attributeNumbers); // traingDataAccuracy
        double testAccuracyAfterPruning = GetDataAccuracy(root, testData, attributeNumbers); // traingDataAccuracy

        System.out.println("\n");
        System.out.println("Post-Pruned Accuracy\n--------------------------------");
        System.out.println("Number of training instances = " + trainingData.size());
        System.out.println("Number of training attributes = " + (trainingData.get(0).size()-1));
        System.out.println("Total number of nodes in the tree = " + nodesInTreeAfterPruning);
        System.out.println("Number of leaf nodes in the tree = " + leafNodesInTreeAfterPruning);
        System.out.println("Accuracy of the model on traianing dataset = " + trainingAccuracyAfterPruning);
        System.out.println("\nNumber of validation instances = " + (validationData.size()));
        System.out.println("Number of validation attributes = " + (validationData.get(0).size()-1));
        System.out.println("Accuracy of the model on validation dataset before pruning = " + validationAccuracyAfterPruning);
        System.out.println("\nNumber of testing instances = " + (testData.size()));
        System.out.println("Number of testing attributes = " + (testData.get(0).size()-1));
        System.out.println("Accuracy of the model on testing dataset = " + testAccuracyAfterPruning);
                
	}
}
