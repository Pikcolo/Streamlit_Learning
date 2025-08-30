# Streamlit_Learning

## Overview

Streamlit_Learning is an interactive web application for image processing, built with [Streamlit](https://streamlit.io/). Users can upload images, capture photos via webcam, or provide image URLs, then apply various processing modes such as Grayscale, Blur, Canny Edge Detection, Thresholding, and custom mixes. The app provides real-time previews, image analysis (histograms and statistics), and export options.

Created by : Sirawich Noipha 6610110327 

## Features

- **Image Input:** Upload, webcam capture, or URL.
- **Processing Modes:** Grayscale, Blur, Canny Edges, Threshold, Custom Mix.
- **Advanced Settings:** Adjustable kernel size, thresholds, and resizing.
- **Analysis:** Histogram visualization and image statistics.
- **Export:** Download processed images in PNG, JPEG, or BMP formats.

## Requirements

See [requirements.txt](requirements.txt) for dependencies:
- streamlit
- opencv-python
- Pillow
- numpy
- matplotlib
- requests

## Usage

### 1. Clone the repository

```sh
git clone https://github.com/Pikcolo/Streamlit_Learning.git
cd Streamlit_Learning
```

### 2. Create and activate a virtual environment

#### Windows (CMD/PowerShell)
```sh
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Run the app
```sh
streamlit run app.py
```

### 5. Access the app

After running the command above, Streamlit will display a local URL such as:

```
Local URL: http://localhost:8501
```

Open this URL in your web browser to use the application.
