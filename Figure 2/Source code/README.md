There are four files in the current folder, corresponding to the experiments in Figure 1 of the paper.
First, you need to execute the "AlexNet image predict.py" file to perform an initial classification of the "(a) input image X". 
Second, you need to execute the "targeted attack.py" file to conduct a targeted attack on the image, generating the adversarial example "(b) attack image X' ". 
Third, you need to execute the "Create perturbing data images of the original and tampered images.py" file to generate the perturbation information image between the input image and the adversarial example.
Finally, the file "output class indices.json" is the index file of the model, which should be imported when needed.
