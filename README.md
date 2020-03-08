# CloudDetection-MODIS
Gaurav Krishnan, Pranshu Chaturvedi, Rohan Prasad, Rutu Patel, Shashank Mahesh, Steven Zheng
NCSA Nvidia Hackathon III

Our Model aims to implement binary classification on specific pixels contained in the MODIS Cloud Dataset to determine the existence of a cloud. Criteria such as band, cloud certainty, and pixel orientation were all considered in creating a suitable dataset for our model. 

# Key Takeaways:

  - Filters cloud files into usable images
  - Creates 4d array of images, bands, and each pixel
  - Performs dimensionality reduction on dataset in order to convert 4d tuple to 2d dataframe
  - Runs model on desired bands
  - Accuracy:  
    - KNN (n=6) = 92.847%
    - SVM = 85.5%
    - Logistic Regression with 3 features = 93.102%
    - Logistic Regression with all features = 95.2% (Possible overfitting)
    - Logistic Regression with pre-determined suitable features = 94.46%
        



