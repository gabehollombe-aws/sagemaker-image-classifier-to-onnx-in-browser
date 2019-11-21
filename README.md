# Train your own custom image classifier and run it at the edge, in the browser, with no network connection!

This repo contains:

1.  A Jupyter notebook suitable for loading into Amazon Sagemaker to train a custom image classifier model and download that model in ONNX format
2.  A web frontend to load an image classifier model trained and downloaded (using the above notebook) and run inferences against that model directly in the browser

## How to use this stuff

1. Start an Amazon Sagemaker Jupyter Notebook instance
2. Upload `sagemaker/train_and_export_as_onnx.ipynb` to your notebook instance
3. Examine the notebook, run it as-is or customize it to download/use your own collection of classified images (one class of images per folder)
4. After running the notebook, download the ONNX model and copy the list of class labels to your clipboard
5. Run `npm install && npm start` to install the web app dependencies and start the single-page-app in your browser
6. Select your ONNX model and paste your list of class labels into the webpage
7. Capture some webcam images, drag some from your filesystem, or paste URLs and watch the inference happen! (The first one takes a few moments to load the model, and subsequent images will infer pretty quickly)
