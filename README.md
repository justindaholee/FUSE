<h1>
  <img src=".\logo\fluora_nobg.png" alt="Your Alt Text" height="25">
  <b>FLUORA</b>: Fluorescent Cell Labeling and Analysis
</h1>

Specialized Pipeline for Cell Segmentation and ROI identification in Time-Series Data
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

<p float="left">
  <img src="https://res.cloudinary.com/apideck/image/upload/v1615737977/icons/google-colab.png" width="100" />
  <img src="./logo/fluora_with_text.png" width="500" /> 
</p>

The Berndt lab developed a cloud-based software called FLUORA, designed to label and analyze fluorescent cells in time-series microscopy images. FLUORA utilizes the Cellpose cell segmentation algorithm and a novel specialized cell alignment algorithm developed by the Berndt lab. With a user-friendly interface through Google Colab, FLUORA allows users to efficiently analyze their data, providing a convenient, free, and fast method for analyzing timecourse data.

## Table of Contents
- [Motivations](#motivations)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Customization](#customization)
- [Results and Output](#results-and-output)
- [Keywords](#keywords)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Motivations
- Population analysis-based platform for fluorescence data
- Generates easy-to-use platform 
- Aligns methods used to analyze fluorescent data
- Novel frame-by-frame cell identification algorithm that can be used with Cellpose segmentation 
- Open source + Free platform
- Mitigating experimental burden associated with hand-drawn segmentation
- Cloud based storage and handling of data attenuates issues with data management and storage

## Features
- Cloud-based implementation using Google Colab
- User-friendly interface with Google Colab forms
- Customizable pipeline for advanced users
- Integration with Cellpose for automated segmentation
- Convolutional autoencoder for feature extraction
- Analysis of fluorescence readouts (delta f) over time

## Prerequisites
- Google Account for accessing Google Colab
- Cellpose-generated segmentation masks
- Fluorescence microscopy videos in TIF format (100-200 frames)

## Getting Started
1. Clone or download the FLUORA repository to your local machine.
2. Upload the repository to your Google Drive.
3. Open the main FLUORA notebook in Google Colab.
4. Set up the Google Colab environment with the required dependencies.

## Customization
Advanced users can modify the underlying code to better suit their specific cell types or experiment requirements. Customization options include adjusting the convolutional autoencoder parameters and modifying the tracking algorithm.

## Results and Output
FLUORA outputs the delta f (change in fluorescence) of the tracked cells over time, providing valuable insights into cell signaling events.

## Keywords: 
in vitro analysis algorithm, transparent data handling, high-throughput, unbiased, user-friendly, visual phenotyping

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

Copyright (c) 2023 University of Washington Department of Bioengineering

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments
- Acknowledge the authors, contributors, and any relevant institutions.
- Acknowledge any funding or support received for the project.
- Mention any third-party libraries or resources used in the project.
