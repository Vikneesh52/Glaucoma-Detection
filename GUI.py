import tkinter as tk
import os
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100, 100), Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    # Update the count based on the prediction
    global num_affected, num_healthy
    if prediction > 0.5:
        num_healthy += 1
    else:
        num_affected += 1
    
    return prediction

def open_folder():
    global file_paths, current_index, is_affected

    folder_path = filedialog.askdirectory()
    if folder_path:
        # Reset the affected region display
        cropped_image_label.config(image='')

        # Get the list of image files in the selected folder
        file_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    file_paths.append(os.path.join(root, file))

        if len(file_paths) > 0:
            current_index = 0
            show_image()
            update_filename_list()

def show_image():
    global current_index, is_affected

    file_path = file_paths[current_index]

    # Reset the affected region display
    cropped_image_label.config(image='')

    image = Image.open(file_path)
    image.thumbnail((400, 400))
    image_tk = ImageTk.PhotoImage(image)
    image_label.config(image=image_tk)
    image_label.image = image_tk

    prediction = import_and_predict(image, model)
    pred = prediction[0][0]

    if pred > 0.5:
        result_label.config(text="Prediction: Your eye is healthy. Great!")
        balloon_label.config(text="")
        # Reset the affected region flag
        is_affected = False
    else:
        result_label.config(
            text="Prediction: You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.")
        balloon_label.config(text="Analyzing affected region...")
        # Set the affected region flag
        is_affected = True
        # Crop and display the affected region
        cropped_image = crop_glaucoma_region(image)
        if cropped_image is not None:
            cropped_image_tk = ImageTk.PhotoImage(cropped_image)
            cropped_image_label.config(image=cropped_image_tk)
            cropped_image_label.image = cropped_image_tk
            cropped_image_label.pack(side=tk.RIGHT, padx=10)

def update_filename_list():
    filename_list.delete(0, tk.END)
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        filename_list.insert(tk.END, filename)

def select_image(event):
    global current_index
    current_index = filename_list.curselection()[0]
    show_image()

def next_image():
    global current_index
    if current_index < len(file_paths) - 1:
        current_index += 1
        show_image()

def previous_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_image()



def autoroi(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = img[y:y + h, x:x + w]

    return roi

def crop_glaucoma_region(image):
    if not is_affected:
        return None

    image = image.convert('RGB')
    image_array = np.array(image)
    image_array = image_array.astype(np.uint8)

    # Use the autoroi function to get the affected region
    affected_region = autoroi(image_array)

    # Create a PIL Image from the cropped affected region
    cropped_image = Image.fromarray(affected_region)

    return cropped_image


def generate_report():
    # Data for the bar chart
    categories = ['Affected', 'Healthy']
    counts = [num_affected, num_healthy]

    # Create the bar chart
    plt.bar(categories, counts)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Glaucoma Cases')
    plt.show()

# Load the trained model
model = tf.keras.models.load_model('glaucoma_model.h5')

# Create the Tkinter window
window = tk.Tk()
window.title("Glaucoma Detector")

# Disable automatic window resizing
window.pack_propagate(0)

# Create the widgets
filename_list = tk.Listbox(window, width=50, height=10)
filename_list.pack(side=tk.LEFT, padx=10, pady=10)
filename_list.bind("<<ListboxSelect>>", select_image)

title_label = tk.Label(window, text="Glaucoma Detector", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10, anchor='center')

instruction_label = tk.Label(window, text="Please upload a JPEG image file", font=("Helvetica", 12))
instruction_label.pack(pady=10,anchor="center")

open_button = tk.Button(window, text="Open Folder", command=open_folder)
open_button.pack(side=tk.RIGHT, pady=10)

next_button = tk.Button(window, text="Next Image", command=next_image)
next_button.pack(side=tk.RIGHT, padx=10)

previous_button = tk.Button(window, text="Previous Image", command=previous_image)
previous_button.pack(side=tk.RIGHT, padx=10)

report_button = tk.Button(window, text="Generate Report", command=generate_report)
report_button.pack(side=tk.RIGHT, pady=10)

image_label = tk.Label(window, width=50, height=20)
image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

result_label = tk.Label(window, text="")
result_label.pack(pady=10)

balloon_label = tk.Label(window, text="")
balloon_label.pack()

cropped_image_label = tk.Label(window)
cropped_image_label.pack(fill=tk.BOTH, expand=True)


# Global variables to track affected and healthy counts
num_affected = 0
num_healthy = 0

# Global variable to track affected region
is_affected = False

# Run the Tkinter event loop
window.mainloop()
