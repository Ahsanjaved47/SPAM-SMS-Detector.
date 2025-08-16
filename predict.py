import joblib
import csv

MODEL_PATH = "models/spam_model.joblib"
OUTPUT_CSV = "predictions.csv"

# Load the trained model
print("[predict] loading model...")
model = joblib.load(MODEL_PATH)
print("[predict] model loaded ✅")

predictions = []

print("Enter SMS messages (type 'done' when finished):")
while True:
    sms = input("SMS: ").strip()
    if sms.lower() == "done":
        break
    if sms == "":
        continue
    label = model.predict([sms])[0]
    print(f"[predict] This message is: {label}")
    predictions.append({"sms": sms, "prediction": label})

# Save predictions to CSV
if predictions:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sms", "prediction"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"[predict] Predictions saved to {OUTPUT_CSV}")
else:
    print("[predict] No SMS messages were entered. Nothing saved.")
