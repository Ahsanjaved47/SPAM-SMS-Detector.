import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import joblib
import csv

MODEL_PATH = "models/spam_model.joblib"
model = joblib.load(MODEL_PATH)

# Store last results for CSV export
last_results = []

# --- Functions ---
def classify_sms():
    global last_results
    text_content = text_input.get("1.0", tk.END).strip()
    if not text_content:
        messagebox.showwarning("Warning", "Please enter some SMS messages!")
        return

    sms_list = text_content.splitlines()
    results = []
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)

    for sms in sms_list:
        sms = sms.strip()
        if sms:
            label = model.predict([sms])[0]
            results.append({"sms": sms, "prediction": label})
            color = "red" if label == "spam" else "green"
            output_text.insert(tk.END, f"{sms} → {label}\n", color)

    output_text.config(state=tk.DISABLED)
    last_results = results

def save_csv():
    if not last_results:
        messagebox.showwarning("Warning", "No predictions to save!")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save predictions as"
    )
    if file_path:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sms", "prediction"])
            writer.writeheader()
            writer.writerows(last_results)
        messagebox.showinfo("Saved", f"Predictions saved to {file_path}")

# --- GUI Setup ---
root = tk.Tk()
root.title("📩 SMS Spam Detector")
root.geometry("700x500")
root.resizable(False, False)
root.configure(bg="#f0f4f7")

# Input Frame
input_frame = tk.Frame(root, bg="#f0f4f7")
input_frame.pack(pady=10)
tk.Label(input_frame, text="Enter SMS messages (one per line):", bg="#f0f4f7", font=("Arial", 12)).pack(anchor="w")
text_input = scrolledtext.ScrolledText(input_frame, width=80, height=10, font=("Arial", 11))
text_input.pack()

# Buttons
btn_frame = tk.Frame(root, bg="#f0f4f7")
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Check Spam", bg="#4CAF50", fg="white", font=("Arial", 12), command=classify_sms).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Save CSV", bg="#2196F3", fg="white", font=("Arial", 12), command=save_csv).grid(row=0, column=1, padx=10)

# Output Frame
output_frame = tk.Frame(root, bg="#f0f4f7")
output_frame.pack(pady=10)
tk.Label(output_frame, text="Results:", bg="#f0f4f7", font=("Arial", 12)).pack(anchor="w")
output_text = scrolledtext.ScrolledText(output_frame, width=80, height=10, font=("Arial", 11), state=tk.DISABLED)
output_text.tag_config("spam", foreground="red")
output_text.tag_config("ham", foreground="green")
output_text.pack()

root.mainloop()
