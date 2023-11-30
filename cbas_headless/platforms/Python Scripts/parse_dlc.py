import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

def splineFilter(df, points):
    x = df.index.tolist()
    for point in points:
        # Create a CubicSpline object for x and y columns
        cs_x = CubicSpline(x, df[(point, 'x')].values)
        cs_y = CubicSpline(x, df[(point, 'y')].values)
        
        # Interpolate the values and store them back in the DataFrame
        df[(point, 'x')] = cs_x(x)
        df[(point, 'y')] = cs_y(x)
    


#add column with velocity of prev to current frame
#Legacy Code for velocity function:
#newCol = []
#newCol = np.sqrt((uniquePart['x'].diff()**2) + (uniquePart['y'].diff()**2))
#newCol.fillna(0, inplace=True)
#
#indx = df.columns.get_loc((point, 'likelihood'))
#df.insert(indx, (point, 'velocity'), newCol)
def velocity(df, points):
    for point in points:
        df[(point, 'velocity')] = np.sqrt((df[(point, 'x')].diff()**2) + (df[(point, 'y')].diff()**2))
        df[(point, 'vel likelihood')] = (df[(point, 'likelihood')] + df[(point, 'likelihood')].shift(1)) / 2
    df.fillna(0, inplace=True)

#Build dict of distances from body parts to fixed points
def distances(df, parts, fixedPoints):
    pointsDict = {}
    for point in fixedPoints:
        pointsDict[point] = {}
        for part in parts:
            pointsDict[point][part] =  []
            pointsDict[point][part].append(np.sqrt((df[(point, 'x')] - df[(part, 'x')])**2 + (df[(point, 'x')] - df[(part, 'y')])**2))
            pointsDict[point][part].append(df[(part, 'likelihood')] * df[(point, 'likelihood')])
    return pointsDict

#Builds dictionary similar to distances func but calculates speed of each part to or from
#each fixed point between the prev and current frame. Requires timeframe in seconds (10fps = 0.1)
def speed(distDict, timeframe):
    speedDict = {}
    for point in distDict:
        speedDict[point] = {}
        for part in distDict[point]:
            speedDict[point][part] = []
            speedDict[point][part].append([
                (distDict[point][part][0][i] - distDict[point][part][0][i -1]) / timeframe 
                if i > 0 else 0 
                for i in range(len(distDict[point][part]))
            ])
            speedDict[point][part].append([
                (distDict[point][part][1][i] + distDict[point][part][1][i -1]) / 2 
                if i > 0 else 0 
                for i in range(len(distDict[point][part]))
            ])
            
    return speedDict


#This can be optimized using to_dict() method. Issue is multiple header rows, df will need to be reshaped first
def dfToDict(df, points):
    dfDict = {}
    for point in points:
        dfDict[point] = {}
        uniquePart = df.loc[:,point]
        for index in uniquePart:
            dfDict[point][index] = uniquePart[index].to_list()
    return dfDict


##################################################################
############################### MAIN #############################
def main():

    #Filename needs standardization or user input
    df = pd.read_csv('dlc_test_file.csv', header=[1,2])
    df.drop(('bodyparts', 'coords'), axis= 1, inplace=True)
    smallDf = df.iloc[::5].copy()
    smallDf.interpolate(method='spline', order=3, inplace=True)
    
    points = []
    for col in df.columns:
        if col[0] not in points:
            points.append(col[0])

    splineFilter(df, points)

    print(smallDf.head())
    splineFilter(smallDf, points)

    print(smallDf.head())

    #These need to be user entered and must match all the columns in the CSV exactly!
    fixedPoints = ['nest', 'spout', 'food_hopper']
    bodyParts = ['nose', 'hand_left', 'hand_right', 'back', 'base_tail']
    

    velocity(df, bodyParts)
    
    distDict = distances(df, bodyParts, fixedPoints)
    speedDict = speed(distDict, 0.1)
    dfDict = dfToDict(df, points)
    


if __name__ == "__main__":
    main()