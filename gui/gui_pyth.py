import tkinter as tk
from tkinter import filedialog, messagebox
from CTkMessagebox import CTkMessagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from customtkinter import *
from PIL import Image

app = CTk()
app.geometry("500x500")
app.resizable(0,0)

# Load the pre-trained PyTorch model
model_path = "C:/Users/Sahlah Baby/Documents/MY STUFF/KU/Spring 2024/Intro to Ai/gui_py/resenet4.pt"
image_folder = "C:/Users/Sahlah Baby/Documents/MY STUFF/KU/Spring 2024/Intro to Ai/gui_py/images"
class_labels = ['No Cancer', 'Cancer']  # 0 is Normal and 1 is Cancer

side_img_data = Image.open("C:/Users/Sahlah Baby/Documents/MY STUFF/KU/Spring 2024/Intro to Ai/gui_py/gui_images/trial2.png")
side_img = CTkImage(dark_image=side_img_data, light_image=side_img_data, size=(500, 300))


def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224))  # Resize the image to match model input size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(img)
        prediction = torch.argmax(output, dim=1).item()
        predicted_class = class_labels[prediction]
        accuracy = torch.softmax(output, dim=1)[0][prediction].item() * 100

        messagebox = CTkMessagebox(title = "Results",message=f"The model predicts: {predicted_class} with {accuracy:.2f}% accuracy",
                                   icon="check",
                                   option_1="OK")
       

model = models.resnet50()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)  # Modify the last layer for your number of classes
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
###################################################################

Label1 = CTkLabel(master=app, text="", image=side_img)
Label1.pack(expand=True, side="top")

frame = CTkFrame(master=app, width=500, height=200, fg_color="#ffffff")
frame.pack_propagate(0)
frame.pack(expand=True, side="bottom")

CTkLabel(master=frame, text="Cancer Diagnosis App", text_color="#601E88", anchor="w", font=("Arial Bold", 24)).pack(anchor="w", pady=(50, 5), padx=(120, 0))
CTkLabel(master=frame, text="Please Uplaod an Image", text_color="#7E7E7E", anchor="w", font=("Arial Bold", 12)).pack(anchor="w", padx=(180, 0))


load_button = CTkButton(frame, text="Load Image and Predict", command=load_and_predict_image, hover_color="#E44982", font=("Arial Bold", 12), text_color="#ffffff", width=225)
load_button.pack(anchor="w", pady=(20, 0), padx=(140, 0))

app.mainloop()