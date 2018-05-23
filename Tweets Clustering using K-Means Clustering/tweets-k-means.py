import json
import sys
import collections

def findClusters(numberOfClusters, initialSeedsFile,tweetsDataFile,outputFile):
    data = readJsonFile(tweetsDataFile)
    seeds = readInitialSeeds(initialSeedsFile,numberOfClusters)
    iterations = 0
    while(True):
        clusters = {}
        for key, value in data.items():
            min = 2 #To find minimum distance making initial distance as 2
            minIndex = 0
            for i in range(len(seeds)):
                centroidString = data[seeds[i]]
                centroidStringWords = centroidString.split()
                dataWords = value.split()
                distance = calculateDistance(centroidStringWords,dataWords)
                if(min>distance):
                    min = distance
                    minIndex = i
            if(seeds[minIndex] not in clusters.keys()):
                lst = []
                lst.append(key)
                clusters[seeds[minIndex]] = lst
            else:
                lst = clusters[seeds[minIndex]]
                lst.append(key)
        newSeeds = findNewCentroids(clusters,data)
        if(collections.Counter(newSeeds) == collections.Counter(seeds)):
            break
        else:
            seeds = newSeeds
        sse = calculateSSE(clusters,data)
        writeClustersToFile(clusters,outputFile,sse)

def calculateSSE(clusters,data):
    sum = 0.0
    for key,value in clusters.items():
        for tweet in value:
            dist =  calculateDistance(data[key].split(),data[tweet].split())
            sum = sum + (dist*dist)
    return sum
        
def writeClustersToFile(clusters,outputFile,sse):
    file = open(outputFile,"w+")
    clusterId = 1
    file.write('SSE Value: ')
    file.write(str(sse))
    file.write("\n")
    for key, value in clusters.items():
        file.write(str(clusterId))
        file.write("  ")
        file.write(str(value).replace("["," ").replace("]",""))
        file.write("\n")
        clusterId = clusterId+1
    
def calculateDistance(text1, text2):
    intersect = set(text1).intersection( set(text2) )
    distinct = set(text1).symmetric_difference( set(text2) )
    distance = 1- (len(intersect)/(len(distinct)+len(intersect)))
    return distance
    
def findNewCentroids(clusters,data):
    centroids = []
    for key, value in clusters.items(): 
        sum = 0
        min = sys.maxsize
        minIndex = 0
        for i in range(len(value)):
            for j in range(len(value)):
                sum+= calculateDistance(data[key].split(),data[value[j]].split())
            if(min>sum):
                min = sum
                minIndex = i
        centroids.append(value[minIndex])
    return centroids
    
def readInitialSeeds(initialSeedsFile,numberOfClusters):
    seeds = []
    lineNumber = 1
    with open(initialSeedsFile) as f:
        for line in f:
            if(lineNumber<=numberOfClusters):
                seeds.append(int(line.replace(",","").strip("\n")))
                lineNumber= lineNumber+1
    return seeds

def readJsonFile(jsonFile):
    data ={}
    with open(jsonFile) as f:  
        #jsonData = json.load(json_file)
        for line in f:
            item = json.loads(line)
            data[item['id']] = item['text']
    return data

if __name__ == "__main__" :
    if (len(sys.argv) != 5):
         print("wrong input. please check readme")
    else:
        numberOfClusters = int(sys.argv[1])
        initialSeedsFile = sys.argv[2]
        tweetsDataFile = sys.argv[3]
        outputFile = sys.argv[4]
    clusters = findClusters(numberOfClusters, initialSeedsFile,tweetsDataFile,outputFile)