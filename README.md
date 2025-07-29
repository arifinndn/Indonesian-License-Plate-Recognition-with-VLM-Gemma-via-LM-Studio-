# Indonesian License Plate Recognition with VLM (Gemma via LM Studio)

## Description

### Concept of VLM (Visual Language Model) and Its Application in OCR

A Visual Language Model (VLM) is an AI model capable of understanding and processing both images and text simultaneously.
Examples of VLMs: Gemma, LLaVA, Flamingo, MiniGPT-4.
This project uses Visual Language Model to perform OCR (Optical Character Recognition) on Indonesian license plate images.
The Gemma model is run locally using LM Studio, and the prediction results are evaluated using Character Error Rate (CER).

Folder Structure
<pre lang="markdown">
├── Indonesian License Plate Recognition Dataset/
│   └── images/
│       └── test/
│           ├── test001_1.jpg
│           ├── test001_2.jpg
│           └── ...
├── labels.csv #groundtruth
├── OCR_license_plate.py #Script for OCR
├── prediction_output # Final result CER score
</pre>
‎
---
## Tools
	1. LM Studio (GUI + API server for Gemma)
	2. Python 3.11
	3. Libraries (pandas, requests, jiwer, Pillow)

---
## How to Run
VLM on LM Studio
Activate the Gemma Model in LM Studio
	
 	1. Open LM Studio.
	2. Click the "Developer" tab on the left side.
	3. Select a model to load (Gemma model)
	4. Enable the "Serve on Local Network" option.
	5. Note the server port (e.g., 1234).
	6. Click the "Status" tab and make sure the model status shows Running.

---
## Dataset
Dataset & Label

	1. Download Dataset (Kaggle)(https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)


	2. Dataset for this project is Indonesian License Plate Recognition (folder test) #197 Photos
		./Indonesian License Plate Recognition Dataset/images/test/

	3. Make sure labels.csv located in the main directory and has the following format:
		image,ground_truth
		image_001.jpg,B1234XYZ
		image_002.jpg,D5678ABC

---
## OCR Script
Running python OCR_license_plate.py

The program will:

	1. Read each image in dataset folder
	2. Send images to the VLM model via LM Studio API
	3. Receive license plate predictions
	4. Calculate Character Error Rate (CER) according to labels.csv as the ground truth
	5. Save results to prediction_output.csv file

---
## Evaluation with CER (Character Error Rate)
To evaluate the prediction quality, the Character Error Rate is used:

CER = (S + D + I) / N

Where:

	S = Number of substitutions (wrong characters)
	D = Deletions (missing characters)
	I = Insertions (extra characters)
	N = Number of characters in the ground truth

A lower CER means better accuracy. CER = 0 means a perfect match.


---
## Output
The output is saved in:
	prediction_output.csv
	
	With the format:
		image,ground_truth,prediction,CER_score
		test001_1.jpg,B1234XYZ,B1234XYZ,0.0
		test001_2.jpg,D5678ABC,D567ABC,0.143

---
## Explaining CER Results: Success vs. Failure

### 1. Successful Example (CER = 0.000)

| Ground Truth     | Prediction     | CER      |
|--------------|--------------|--------------|
| B1234XYZ | B1234XYZ | 0.0 |

Explanation:
Every character matches
No substitutions, deletions, or insertions


### 2. Failed Example (CER = 0.375)

Ground Truth	Prediction	CER
		
| Ground Truth     | Prediction     | CER      |
|--------------|--------------|--------------|
| B1234XYZ | B124XYZ | 0.25 |

Calculation:
Ground truth has 8 characters

1 substitution: ‘3’ replaced by ‘4’

1 deletion: missing character ‘4’

CER = (1 + 1 + 0) / 8 = 0.25

Explanation:
The VLM may have misread one digit. Possibly due to image noise, blur, or an angled plate
