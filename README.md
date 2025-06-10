<h1>
  <b>FUSE</b>: Fluorescent Signal Engine
</h1>

Specialized Pipeline for Cell Segmentation and Alignment of Time-Series Data
<a target="_blank" href="https://colab.research.google.com/github/justindaholee/FUSE/blob/main/FUSE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

The Berndt lab developed a cloud-based software called FUSE, designed to segment and align fluorescent cells in time-series microscopy images. FUSE utilizes the Cellpose cell segmentation algorithm and a novel specialized cell alignment algorithm developed by the Berndt lab. With a user-friendly interface through Google Colab, FUSE allows users to efficiently analyze their data, providing a convenient, free, and fast method for analyzing timecourse data.

<p float="left">
  <img src="https://res.cloudinary.com/apideck/image/upload/v1615737977/icons/google-colab.png" width="120" />
  <img src="fuse.png" width="250">
</p>

## Table of Contents
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Results and Output](#results-and-output)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Key Features
- Population analysis-based approach for cell-image alignment and data simplification, establishing a standardized framework for analyzing fluorescent cell time-series images.
- Utilizes a novel autoencoder-powered, frame-by-frame cell alignment algorithm compatible with Cellpose segmentation.
- Offers cloud-based data storage and management through Google Colab and Google Drive.
- Supports customization and automated segmentation of time-series images with Cellpose.
- Quick generation and preview of various fluorescent signals, including fluorescent traces and ratiometric signals (with more options in development).

## Prerequisites
- Google Account for accessing Google Colab
- Fluorescence microscopy videos in TIF format stored in Google Drive

## Getting Started

Welcome to the `fuse-toolkit` package! Getting started is easy, and you have two primary options to choose from:

### Option 1: Interactive Analysis with Google Colab

1. **Open the Colab Notebook**: Access our interactive notebook by clicking the "Open In Colab" badge at the top of this README. Alternatively, you can use [this link](https://colab.research.google.com/github/justindaholee/FUSE/blob/main/FUSE.ipynb).

2. **Create Your Copy**: In the Colab notebook, make your copy by selecting "File" in the top menu and then "Save a copy in Drive." This ensures you have an editable version.

3. **Start Analyzing**: Follow the step-by-step instructions in the notebook to perform your analysis with the `fuse-toolkit` package. It offers a user-friendly, code-free interface.

### Option 2: Local Installation and Customization

1. **Installation**: If you prefer working in your local environment, install the `fuse-toolkit` package via pip:

   ```shell
   pip install fuse-toolkit

2. **Explore the Documentation**: Discover the full potential of the toolkit in our comprehensive [documentation](https://github.com/shanizu/FUSE). It provides detailed information on installation, usage, and customization.

3. **Code Integration**: Import the `fuse-toolkit` package into your Python environment. Use it to tailor your analysis according to your specific requirements. The documentation is a valuable resource, providing examples and best practices.

Choose the method that suits your workflow best, and unlock the power of the fuse-toolkit for your time-series fluorescent cell data analysis.

## Results and Output

When you use the `fuse-toolkit`, you can expect a set of valuable output data and insights that enhance your analysis of time-series microscopy images. The key outputs of the toolkit include:

1. **Cell Segmentation Masks**: The toolkit generates high-quality Cellpose masks, enabling accurate cell segmentation in your images. These masks provide the foundation for further analysis and visualization.

2. **CSV Files with Aligned Cell ROI Information**: For each frame in your time-series data, the toolkit produces CSV files containing information about the regions of interest (ROI) for each cell. These files are meticulously aligned, making it easier to track individual cell behavior across frames.

3. **Generated Signals**: The toolkit can produce various signals based on your fluorescence data. These signals include fluorescent traces and ratiometric signals, providing critical insights into cell signaling events. Further options for signal generation are under development, enhancing your analytical capabilities.

With the `fuse-toolkit`, you gain a comprehensive set of data and insights, enabling you to dive deep into your time-series data, track cell behavior, and extract valuable information for your research or experiments.

Feel free to explore the generated outputs and integrate them into your analysis workflow to unlock the full potential of your data.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

Copyright (c) 2023 University of Washington Department of Bioengineering

## Acknowledgments
- Justin Daho Lee, Sarah Wait, Aida Moghadasi, Andre Berndt, PhD.
- University of Washington Department of Bioengineering, Mary Gates Research Endowment
- [Cellpose](https://github.com/MouseLand/cellpose)

### Dependency Notice
FUSE relies on the Cellpose cell segmentation algorithm, an open-source tool provided by the Howard Hughes Medical Institute. Cellpose is subject to its own license, which can be found in the [LICENSE](https://github.com/MouseLand/cellpose/blob/main/LICENSE) file of the Cellpose repository. Please review and comply with Cellpose's licensing terms when using this software.
