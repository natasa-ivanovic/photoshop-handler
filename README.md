# Photoshop Handler - Soft Computing project

## Running the project:  
Install all the requirements listed in the requirements.txt (preferably use a venv).  
You must have Photoshop installed, preferably a version equal to or later than Photoshop CC2015.  
Before starting the project, configure Photoshop brush and delete tool to be of same size (30 works okay with 1920x1080 resolution) and close the program before launching.  
Check if the project contains 'best_model.hdf5' file in root (it should, if it doesn't you can get it from [this link](https://www.dropbox.com/s/qs5xzpuk6qv2x7m/best_model.hdf5?dl=0)).  
Launch main_module.py and make sure that save_images is set to False.  
## Options  
If you would like to see the webcam results, you can set debug_mode to True.  
If you would like to flip the view of the camera, you can set flip_coefficient to -1.  
If using the debug mode, you must press B to capture the background when you're ready to do so. If not in debug mode the background is captured on app launch.  
## Training the neural network
If you would like to train the network again (or use your own dataset), create and populate a folder 'dataset' with [our dataset](https://www.dropbox.com/s/f9e4pl2qllit9qo/dataset.zip?dl=0), or create your own dataset!  

## Creating your own dataset  
1. Set debug_mode to True and save_images to True  
2. Set gesture_name to one of 6 supported gestures (brush, nav, zoom, delete, move or pan)
3. Set your image counter start and limit (with params img_counter and final_img_count)  
4. Launch the app and use spacebar or click on the window titled 'original' to capture an image! 

### Team members:  
Nataša Ivanović - SW47/2017  
Mario Kujundžić - SW59/2017  
