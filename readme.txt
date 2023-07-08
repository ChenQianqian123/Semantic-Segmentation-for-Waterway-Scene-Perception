1.Detect the lifetimes of keypoints in water and non-water areas respectively --> .LToF dection.py
2.Cluster long-lived keypoints to generate a likelihood function --> .Pixel clustering.py
3.Train the U-Net model --> .unet_train.py
4.Predict with the U-Net model to generate prior probabilities --> .bigmappredict.py
5.Integrate the U-Net model's prediction results with the likelihood function within a Bayesian framework, generate posterior probabilities, and complete semantic segmentation for the water navigation scene --> .Bayesian.py
unet_predictions is a NumPy array with shape (n, height, width), where n is the number of U-Net model prediction results, height and width are the dimensions of the image.
compute_likelihood is a function that takes a U-Net prediction and some other parameters, and returns a likelihood function value.
6.Accuracy evaluation --> .EvaluationMetrics.py