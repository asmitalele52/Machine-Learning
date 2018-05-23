Objective -
Compute the similarity between tweets using the Jaccard Distance metric.
Cluster tweets using the K-means clustering algorithm.

Calculation of Jaccard distance between two tweets -  https://en.wikipedia.org/wiki/Jaccard_index

input -
1. number of clusters
2. A real world dataset sampled from Twitter as json (find a sample - tweets.json)
3. Initial seeds as centroids for clustering (find a sample - initialSeeds.txt)

Output -
a file that contains clusterId and their respective tweets
Also the SSE value is printed in this file

code is implemented in python3.6

Run the program using the command line
Change directory to current working directory and then execute following command.
#initialSeedsFile and TweetsDataFile are provided in the folder for reference
# initialseeds should be greater or equal to number of seeds

python tweets-k-means.py numberOfClusters initialSeedsFile TweetsDataFile outputFile

#python  tweets-k-means.py 25 initialSeeds.txt tweets.json tweets-k-means-output.txt