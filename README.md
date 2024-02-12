


<div style="display: grid; grid-template-rows: repeat(1, 1fr); gap: 10px;">
    <div style="display: flex; flex-direction:row; align-items: center; justify-content: space-around; padding: 50px">
        <img src="./cbas_headless/assets/CBAS_logo.png" alt="CBAS Logo" style="width: 800px; height: auto;">
    </div>
</div>




# CBAS (Circadian Behavioral Analysis Suite)

CBAS is a suite of tools for phenotyping of complex behaviors. It is designed to automate inferencing of complex behaviors from active live streams of video data, and to provide a simple interface for visualizing and analyzing the results. CBAS currently supports automated inferencing from state-of-the-art machine learning vision models including deepethogram and deeplabcut. CBAS also includes a transformer-based sequence model for deepethogram outputs, which is designed to be more accurate and more efficient than the original deepethogram sequence model. 

Written and maintained by Logan Perry in the Jones Lab at Texas A&M University.


<div style="display: grid; grid-template-rows: repeat(1, 1fr); gap: 10px;">
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
        <div style="display: flex; flex-direction:row; align-items: center; justify-content: space-around; padding: 50px">
            <img src="./cbas_headless/assets/main.png" alt="CBAS Diagram" style="width: 800px; height: auto;">
        </div>
        <div style="display: flex; flex-direction:row; align-items: center; justify-content: space-around; padding: 50px">
            <img src="./cbas_headless/assets/realtime.gif" alt="CBAS in action" style="width: 800px; height: auto;">
        </div>
    </div>
</div>

# Installation

A headless version of CBAS is available on PyPI and can be installed using pip:

```pip install cbas_headless```

A gui version of CBAS is currently under construction and will be available soon.


# Usage

The headless version of CBAS is designed to be used in ipython or jupyter notebooks. Jupyter notebooks for examples of how to use CBAS to automate a pre-existing deepethogram or deeplabcut model are currently under construction but will be found below in the near future.

## Video Acquisition and Automatic Inference

CBAS was built with live video streams in mind. The video acquisition module provides a simple interface for acquiring video from real-time network streams via RTSP. RTSP is a widely supported protocol for streaming video over the internet, and is supported by many IP cameras and network video recorders. CBAS uses ffmpeg to acquire video from RTSP streams, and can be used to acquire video from any RTSP source that ffmpeg supports. 

CBAS video acquisition can be used with or without deepethogram or deeplabcut models. When used with a model or multiple models, CBAS can be used to automatically infer the behavior of animals in real-time. CBAS seamlessly handles model context switching, allowing the user to inference video streams with any number or type of vision models. When a recording is finished, CBAS continues inferrencing the video stream until all videos are inferenced.

<div style="display: grid; grid-template-rows: repeat(1, 1fr); gap: 10px;">
    <div style="display: grid; grid-template-columns: repeat(1, 1fr); gap: 10px;">
        <div style="display: flex; flex-direction:row; align-items: center; justify-content: space-around; padding: 50px">
            <img src="./cbas_headless/assets/realtime_inference.gif" alt="CBAS Diagram" style="width: 1000px; height: auto;">
        </div>
    </div>
</div>

## Training Set Creation

The training set creation module allows the user to manually annotate the videos from any number of recordings with the behavior of interest. The annotated videos can then be used to train a deepethogram model or for training a CBAS sequence model for deepethogram outputs. 

## Model Validation

The model validation module allows the user to validate the performance of a deepethogram or CBAS sequence model on naive test sets. The user can use the module to visualize the model's performance on the videos, and to calculate the model's performance metrics (precision, recall, f1-score, balanced accuracy, etc.). 


<div style="display: grid; grid-template-rows: repeat(1, 1fr); gap: 10px;">
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-end;">
            <img src="./cbas_headless/assets/piechart.png" alt="Training Set Pie Chart" style="width: 300px; height: auto;">
        </div>
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-end;">
            <img src="./cbas_headless/assets/prcurve.png" alt="Precision Recall Curve" style="width: 300px; height: auto;">
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-center;">
            <div style="text-align: center; width: 250px;">
                The CBAS validation module provides a pie chart of the behavior distribution in the training set. This is particularly useful for maintaining balanced training sets. 
            </div>
        </div>
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-center;">
            <div style="text-align: center; width: 250px;">
                CBAS validation also provides precision-recall curves for verifying a model's performance on a test set.
            </div>
        </div>
    </div>
</div>

## Visualization and Analysis

The visualization and analysis module provides circadian plotting tools for visualizing and analyzing the behavior of animals over time. Circadian methods of particular importance are actogram generation, behavior transition raster generation, timeseries fitting and circadian parameter extraction with CosinorPy integration, and timeseries exportation for ClockLab Analysis.


<div style="display: grid; grid-template-rows: repeat(1, 1fr); gap: 10px;">
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-end; padding: 10px;">
            <img src="./cbas_headless/assets/actogram.png" alt="Population Average Actogram" style="width: 300px; height: auto;">
        </div>
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-end; padding: 10px;">
            <img src="./cbas_headless/assets/raster.png" alt="Population Average Transition Raster" style="width: 600px; height: auto;">
        </div>
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-end; padding: 10px;">
            <img src="./cbas_headless/assets/comparison.png" alt="Average Transition Raster Comparison Across Populations" style="width: 600px; height: auto;">
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-center; padding: 10px;">
            <div style="text-align: center; width: 250px;">
                An example of a population average actogram generated by CBAS. The actogram is generated by binning the behavior of all animals in the population into 30 minute bins, and plotting the average behavior in each bin.
            </div>
        </div>
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-center; padding: 10px;">
            <div style="text-align: center; width: 250px;">
                An example of a population average transition raster generated by CBAS. The transition raster is generated by binning the behavior of all animals in the population into 30 minute bins, and plotting the average behavior transition probability in each bin.
            </div>
        </div>
        <div style="display: flex; flex-direction:column; align-items: center; justify-content: flex-center; padding: 10px;">
            <div style="text-align: center; width: 250px;">
                An example of a comparison of population average transition rasters generated by CBAS. The comparison raster is generated by 'subtracting' two population transition rasters. The comparison in this case is between wild type male mice and wild type female mice.
            </div>
        </div>
    </div>
</div>

# Credits

 - Deepethogram is a state-of-the-art vision model for inferring complex behaviors from video data. It was developed by Jim Bohnslav at Harvard. A few components of the original deepethogram package have been included in the platforms/modified_deepethogram directory. These changes were made to hold the deepethogram model in memory and to allow for real-time inferencing of video streams. Only the necessary components of the original deepethogram package were included in the modified version. Please support the original deepethogram package by visiting their repository, [here](https://github.com/jbohnslav/deepethogram).

 - Deeplabcut is a skeletal based vision model. It was developed by the A. and M.W. Mathis Labs. Please support the original deeplabcut package by visiting their repository, [here](https://github.com/DeepLabCut)

# Acknowledgements



# License
MIT License
