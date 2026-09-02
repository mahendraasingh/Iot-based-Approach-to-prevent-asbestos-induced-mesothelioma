# Asbestos Detection and Mesothelioma Risk Analysis System
# For Google Colab - Run this entire code block

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import datetime
import cv2
import PIL
from PIL import Image
import warnings
import io
import gc
import json
import requests
import random
from google.colab import drive, output, files
import base64
import time
import IPython.display
from ipywidgets import widgets
from IPython.display import display, clear_output, HTML

# Suppress warnings
warnings.filterwarnings('ignore')

# Mount Google Drive for data storage
drive.mount('/content/drive', force_remount=True)

# Create necessary directories
base_dir = '/content/asbestos_detection'
os.makedirs(base_dir, exist_ok=True)
model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
reports_dir = os.path.join(base_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)

print("âœ… Initial setup complete")

# Synthetic data generation for demonstration purposes
def generate_synthetic_data():
    print("Generating synthetic training data...")
    
    # Create directories for synthetic data
    train_dir = os.path.join(data_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'asbestos'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'non_asbestos'), exist_ok=True)
    
    # Generate synthetic asbestos images (white fibrous patterns on dark background)
    for i in range(100):
        img_size = 224
        # Create a dark background
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        img.fill(50)  # Dark gray background
        
        # Add white fibrous structures (typical of asbestos)
        num_fibers = np.random.randint(5, 15)
        for _ in range(num_fibers):
            start_x, start_y = np.random.randint(0, img_size, 2)
            length = np.random.randint(20, 80)
            thickness = np.random.randint(1, 4)
            angle = np.random.rand() * 360
            
            # Calculate end point
            end_x = int(start_x + length * np.cos(np.radians(angle)))
            end_y = int(start_y + length * np.sin(np.radians(angle)))
            
            # Draw the fiber
            cv2.line(img, (start_x, start_y), (end_x, end_y), (200, 200, 200), thickness)
        
        # Add some noise
        noise = np.random.randint(0, 30, size=img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(train_dir, 'asbestos', f'asbestos_{i}.jpg'))
    
    # Generate non-asbestos images (random textures without fibrous patterns)
    for i in range(100):
        img_size = 224
        # Create a textured background
        img = np.random.randint(30, 120, (img_size, img_size, 3), dtype=np.uint8)
        
        # Add some blobs/particles (not fibrous)
        num_particles = np.random.randint(5, 20)
        for _ in range(num_particles):
            center_x, center_y = np.random.randint(0, img_size, 2)
            radius = np.random.randint(3, 10)
            color = tuple(np.random.randint(100, 200, 3).tolist())
            cv2.circle(img, (center_x, center_y), radius, color, -1)
        
        # Add some noise
        noise = np.random.randint(0, 20, size=img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Apply blur to make it look more natural
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(train_dir, 'non_asbestos', f'non_asbestos_{i}.jpg'))
    
    print("âœ… Synthetic data generation complete.")

# Generate synthetic environmental data for time series analysis
def generate_environmental_data():
    print("Generating synthetic environmental data...")
    
    # Create a dataframe with timestamps and environmental factors
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
    
    # Initialize data dictionary
    data = {
        'date': dates,
        'temperature': [],
        'humidity': [],
        'asbestos_concentration': [],
        'exposure_duration': [],
        'ventilation_rate': [],
        'risk_score': []
    }
    
    # Generate data with seasonal patterns and trends
    for i in range(365):
        # Temperature: seasonal pattern (higher in summer)
        temp_base = 22 + 10 * np.sin(2 * np.pi * i / 365)
        temp = temp_base + np.random.normal(0, 2)
        
        # Humidity: inverse correlation with temperature + random noise
        humidity_base = 60 - 0.5 * (temp_base - 22)
        humidity = humidity_base + np.random.normal(0, 5)
        humidity = max(30, min(humidity, 95))
        
        # Asbestos concentration: some correlation with temperature and ventilation
        # Simulating higher concentrations when maintenance work is done (random spikes)
        is_maintenance = random.random() < 0.05  # 5% chance of maintenance day
        base_concentration = 0.1 + 0.05 * np.sin(2 * np.pi * i / 365 + np.pi)  # Seasonal pattern
        spike = 0.4 if is_maintenance else 0
        concentration = base_concentration + spike + np.random.normal(0, 0.03)
        concentration = max(0.01, concentration)
        
        # Ventilation rate: better in newer systems, worse in older
        ventilation = 80 + np.random.normal(0, 5)
        
        # Exposure duration: working hours but with some variation
        exposure = 8 + np.random.normal(0, 0.5)
        
        # Risk score: composite score based on concentration and exposure
        risk = concentration * exposure * (1 - ventilation/100) * 10
        
        # Add to data dictionary
        data['temperature'].append(temp)
        data['humidity'].append(humidity)
        data['asbestos_concentration'].append(concentration)
        data['exposure_duration'].append(exposure)
        data['ventilation_rate'].append(ventilation)
        data['risk_score'].append(risk)
    
    # Create DataFrame
    env_df = pd.DataFrame(data)
    
    # Save to CSV
    env_df.to_csv(os.path.join(data_dir, 'environmental_data.csv'), index=False)
    
    print("âœ… Environmental data generation complete.")
    
    return env_df

