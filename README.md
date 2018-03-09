# Segmentation of Geometrical Shapes in Aerial Images
This is my Master thesis project at ETH Zurich, which contains two parts:
* Buildings Detection,
* Roads Detection (not included yet).

## Thesis Outline
* Introduction
	* Background
	* Focus of This Work
	* Thesis Organization
* Related Work
	* Previous Theses
	* Recent Models
		* PolygonRNN
		* Mask R-CNN
	* Motivation
* Model Architecture
	* PolygonRNN
		* VGG-16
		* ConvLSTM
	* Faster/Mask R-CNN
		* Region Proposal Network
		* Feature Pyramid Network
	* Region-based PolygonRNN
		* Two-step Model
		* Hybrid Model
* Experiments and Results
	* Data Preparation
		* Buildings and Areas
		* OpenStreetMap
		* Google Maps APIs
		* Shift Adjustment
	* Implementation Details
		* Configuration
		* Beam Search
		* Training
	* Experiment Results
		* Single Building Segmentation
		* Buildings Localization
		* Region-based PolygonRNN
* Problems and Future Work
	* Problems
	* Future Work

