I want to start a Python project for analyzing construction floor plans.

**My Goal (Step 1):** Build a script that takes a floor plan PDF, processes it to detect walls (ignoring grid lines), calculates total wall length, and exports a preliminary Bill of Quantities (BoQ) to a CSV file.

**The Input:** A floor plan PDF (often contains grid lines/tiles which are noise).

**The Logic:**
1. Convert PDF to high-res image (using PyMuPDF).
2. Use OpenCV to preprocess: Grayscale -> Threshold -> **Morphological Operations** (kernel size ~5,5) to remove thin grid lines and keep thick walls.
3. Count wall pixels (non-zero pixels in the processed image).
4. Implement a simple calibration variable (e.g., `PIXELS_PER_METER = 50`) to convert pixels to meters.
5. **Output:** Generate a `boq.csv` file with columns: `Item`, `Quantity`, `Unit`. 
   (Example item: 'Wall Construction', Quantity: [calculated meters], Unit: 'm').

**Please create the following files:**
1. `requirements.txt` (include `pymupdf`, `opencv-python-headless`, `pandas`, `numpy`).
2. `analyzer.py`: The class that handles the image processing and calculation logic.
3. `main.py`: A script that runs the analyzer on a hardcoded file named "plan.pdf" and prints the results.