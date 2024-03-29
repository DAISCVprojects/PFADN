# PFADN

This repository contains implementation of "Deep Demosaicing for Polarimetric Filter Array Cameras". If you find this code useful for your research work, cite following:

@article{pistellato2022deep,
  title={Deep demosaicing for polarimetric filter array cameras},
  author={Pistellato, Mara and Bergamasco, Filippo and Fatima, Tehreem and Torsello, Andrea},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={2017--2026},
  year={2022},
  publisher={IEEE}
}

## Requirements

- numpy 1.19.2  
- OpenCV 3.4.2  
- tensorflow 2.1.0  
- h5py 2.8.0

## Pre-trained Weights

There are three weights files inside "models" folder, they are trained on different datasets:

- **PFADN_synth_only:** trained on synthetic data only  
- **PFADN_mitsuba_weights:**  trained on data generated by mitsuba  
- **PFADN_camera_model_14_11:** trained on real world data

## Usage

**test.ipnyb** takes mosaic image as input (make sure pattern of filters is same as one described in paper) and outputs demosaiced intensity and Angle of Linear Polarization (AOLP) image inside **results** folder. You can change weights inside same notebook. 

## NOTE

Above mentioned weights were computed by training model on negative AoLP. If you are considering fine tuning PFADN model on your data, multiply your ground truth AoLP with minus before feeding it to the network in order to maintain consistency. 
