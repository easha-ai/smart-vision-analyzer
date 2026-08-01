# Smart Vision Analyzer

Smart Vision Analyzer is a real-time computer-vision application that combines YOLOv8 object detection with interactive color analysis. It provides a Gradio web interface for viewing camera frames, detecting objects, inspecting colors at selected points, visualizing edges, and exporting color observations.

## Features

- Real-time object detection powered by the YOLOv8 nano model
- Point-and-click RGB color identification
- Persistent crosshair for tracking a selected point
- Human-readable color matching using `colors.csv`
- Optional Canny edge-detection view
- Adjustable object-detection confidence threshold
- Detection statistics and a recent-color palette
- CSV export for captured color data
- Browser-based interface built with Gradio

## Project structure

```text
smart-vision-analyzer/
|-- color_detector1.py   # Main application
|-- colors.csv           # Color names and RGB reference values
|-- yolov8n.pt           # YOLOv8 nano model weights
`-- README.md
```

## Requirements

- Python 3.10 or 3.11 recommended
- A working webcam
- A modern web browser with camera permission enabled

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/easha-ai/smart-vision-analyzer.git
cd smart-vision-analyzer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python numpy pandas gradio ultralytics matplotlib pillow
```

## Running the application

Start the app from the repository directory:

```bash
python color_detector1.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser and allow camera access when prompted.

## How to use

1. Wait for the camera feed and YOLO model to initialize.
2. Adjust the confidence slider to control object-detection sensitivity.
3. Enable **Show Color Detection** and click a point in the processed output.
4. View the crosshair, RGB value, nearest color name, and recent-color palette.
5. Enable edge detection to display object boundaries beside the annotated frame.
6. Use **Clear Selection** to choose another point, or **Reset Statistics** to clear the session data.
7. Use **Save Color Data** to export recent color detections to a timestamped CSV file.

## Troubleshooting

### The camera does not appear

- Allow camera access for your browser and operating system.
- Close other applications that may already be using the webcam.
- Make sure the page is opened from `http://127.0.0.1:7860`.

### PowerShell blocks virtual-environment activation

You can run the virtual environment without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python numpy pandas gradio ultralytics matplotlib pillow
.\.venv\Scripts\python.exe color_detector1.py
```

### The YOLO model fails to load

Confirm that `yolov8n.pt` is present in the same directory as `color_detector1.py` and that the `ultralytics` package installed successfully.

## Technology stack

- [Ultralytics YOLOv8](https://docs.ultralytics.com/) for object detection
- [OpenCV](https://opencv.org/) for image processing and edge detection
- [Gradio](https://www.gradio.app/) for the web interface
- Pandas, NumPy, Matplotlib, and Pillow for data and image utilities

## Notes

- Processing is performed locally on the machine running the application.
- Exported files use the name `color_detections_YYYYMMDD_HHMMSS.csv`.
- Detection performance depends on the available CPU/GPU, camera resolution, and confidence setting.

## Contributing

Contributions and suggestions are welcome. Open an issue or submit a pull request with a clear description of the proposed improvement.
