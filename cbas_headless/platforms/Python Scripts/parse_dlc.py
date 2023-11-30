#Calculate distance from bodypart to object as timeseries
#Mutliply likelyhood of two points together to get a general likelyhood for the distance
#List of lists for distance of every point on the mouse to fixed reference point
#Fixed ref points will be hardcoded list for now, will end up being user input that matches header names

#Build dict of numpy arr
#{
#nose:{x:[...],y:[...]}
#lefthand:{x:[...],y:[...]
#}

import pandas as pd
import numpy as np

def distances(df, parts, fixedPoints):
    pointsDict = {}
    for point in fixedPoints:
        #Need unique part here for fixed point for body part to reference
        pointsDict[point] = []
        for part in parts:
            uniquePart = df.loc[:,part]#this will reference outer for loop unique part, not master df CHANGE THIS

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
    fixedPoints = ['nest', 'spout', 'food_hopper']
    bodyParts = ['nose', 'hand_left', 'hand_right', 'back', 'base_tail']
    

    for col in df.columns:
        if col[0] not in points:
            points.append(col[0])

    velocity(df, points)
    
    distances = distances(df, bodyParts, fixedPoints)

    

    

if __name__ == "__main__":
    main()