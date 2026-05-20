import pandas as pd

df = pd.read_excel("measurements.xlsx")

labels = pd.DataFrame({
    "image_name": df["Num"].astype(str) + ".png",
    "weight": df["Body weight (kg)"],
    "body_length_cm": df["Oblique body length (cm)"],
    "withers_height_cm": df["Withers height(cm)"],
    "heart_girth_cm": df["Heart girth(cm)"],
    "hip_length_cm": df["Hip length (cm)"],
    "bcs": df["BCS"]
})

labels.to_csv("data/labels.csv", index=False)

print("data/labels.csv created")