import requests
import pandas as pd
from jiwer import cer
from PIL import Image
import base64
import os


# Configurasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
DATASET_CSV_PATH = os.path.join(BASE_DIR, "labels.csv")
IMAGE_FOLDER = os.path.join(BASE_DIR, "Indonesian License Plate Recognition Dataset/images/test/")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "prediction_output.csv")


# Konversi gambar ke Base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Request ke LM Studio
def predict_plate(image_path):
    base64_image = encode_image(image_path)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "google/gemma-3-12b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the license plate number in this image? Reply only with the plate number."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }
    response = requests.post(LMSTUDIO_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        print("Error:", response.text)
        return ""



# Main Program
def main():
    df = pd.read_csv(DATASET_CSV_PATH)
    predictions = []
    cer_scores = []

    for idx, row in df.iterrows():
        image_file = IMAGE_FOLDER + row['image']
        gt = row['ground_truth']

        pred = predict_plate(image_file)
        cer_score = cer(gt, pred)

        print(f"{row['image']} | GT: {gt} | Pred: {pred} | CER: {cer_score:.3f}")

        predictions.append(pred)
        cer_scores.append(cer_score)

    # Simpan hasil ke CSV
    df['prediction'] = predictions
    df['CER_score'] = cer_scores

    average_cer = sum(cer_scores) / len(cer_scores)
    print(f"\n📊 Average CER: {average_cer:.3f}")

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n✅ Output saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
