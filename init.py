from data_prep_weights import PlacePulseDatasetWeight
import os
import shutil

if os.path.exists('place-pulse-2.0'):
    reset = input("PlacePulse dataset already exists. Do you want to reset it? (y/n): ")
    if reset.lower() == 'y':
        print("Resetting PlacePulse dataset...")
        shutil.rmtree('place-pulse-2.0')
        PlacePulseDatasetWeight.load()
    else: 
        print("Using existing PlacePulse dataset.")
        PlacePulseDatasetWeight.preprocess()
else:
    PlacePulseDatasetWeight.load()