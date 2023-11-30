#Build dict of numpy arr
#{
#nose:{x:[...],y:[...]}
#lefthand:{x:[...],y:[...]
#}

import pandas as pd
import numpy as np

#Builds dictionary similar to distances func but calculates speed of each part to or from
#each fixed point between the prev and current frame. Requires timeframe in seconds (10fps = 0.1)
def speed(distDict, timeframe):
    speedDict = {}
    for point in distDict:
        speedDict[point] = {}
        for part in distDict[point]:
            speedDict[point][part] = [
                (distDict[point][part][i] - distDict[point][part][i -1]) / timeframe 
                if i > 0 else 0 
                for i in range(len(distDict[point][part]))
            ]
            
    return speedDict

#Build dict of distances from body parts to fixed points
def distances(df, parts, fixedPoints):
    pointsDict = {}
    for point in fixedPoints:
        uniquePoint = df.loc[:,point]
        pointsDict[point] = {}
        for part in parts:
            uniquePart = df.loc[:,part]
            pointsDict[point][part] =  np.sqrt((uniquePoint['x'] - uniquePart['x'])**2 + (uniquePoint['y'] - uniquePoint['y'])**2)
    return pointsDict


#add column with velocity of prev to current frame
def velocity(df, points):
    for point in points:
        newCol = []
        uniquePart = df.loc[:,point]
        newCol = np.sqrt((uniquePart['x'].diff()**2) + (uniquePart['y'].diff()**2))
        newCol.fillna(0, inplace=True)
        
        indx = df.columns.get_loc((point, 'likelihood'))
        df.insert(indx, (point, 'velocity'), newCol)

def main():
    df = pd.read_csv('dlc_test_file.csv', header=[1,2])
    df.drop(('bodyparts', 'coords'), axis= 1, inplace=True)
    
    points = []
    for col in df.columns:
        if col[0] not in points:
            points.append(col[0])


    fixedPoints = ['nest', 'spout', 'food_hopper']
    bodyParts = ['nose', 'hand_left', 'hand_right', 'back', 'base_tail']
    

    velocity(df, points)
    
    distDict = distances(df, bodyParts, fixedPoints)
    speedDict = speed(distDict, 0.1)
    print(distDict)
    
    

    

if __name__ == "__main__":
    main()