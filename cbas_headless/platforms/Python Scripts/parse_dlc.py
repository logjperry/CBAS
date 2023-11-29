#Calculate distance from bodypart to object as timeseries
#Mutliply likelyhood of two points together to get a general likelyhood for the distance

#Build dict of numpy arr
#{
#nose:{x:[...],y:[...]}
#lefthand:{x:[...],y:[...]
#}

import pandas as pd
import numpy as np

#add column with velocity of prev to current frame
def velocity(df, points):
    for point in points:
        uniquePart = df.loc[:,point]
        uniquePart['velocity'] = np.sqrt((uniquePart['x'].diff()**2) + (uniquePart['y'].diff()**2))
        uniquePart['velocity'].fillna(0, inplace=True)
        
        df.loc[:,point]
        print(uniquePart.head())


    return 

def main():
    df = pd.read_csv('dlc_test_file.csv', header=[1,2])
    df.drop(('bodyparts', 'coords'), axis= 1, inplace=True)
    print(df.head())
    
    points = []

    for col in df.columns:
        if col[0] not in points:
            points.append(col[0])

    print(df.loc[:,points[0]])
    velocity(df, points)
    

    

if __name__ == "__main__":
    main()