# Train your own custom image classifier and run it at the edge, in the browser, with no network connection!

This repo contains:

1.  A Jupyter notebook suitable for loading into Amazon Sagemaker to train a custom image classifier model and download that model in ONNX format
2.  A web frontend to load an image classifier model trained and downloaded (using the above notebook) and run inferences against that model directly in the browser

## How to use this stuff

### Train an image classifer model and export it as ONNX
1. Start an Amazon Sagemaker Jupyter Notebook instance
2. Upload `sagemaker/train_and_export_as_onnx.ipynb` to your notebook instance
3. Examine the notebook, run it as-is or customize it to download/use your own collection of classified images (one class of images per folder)
4. After running the notebook, download the ONNX model and copy the list of class labels to your clipboard


### Load the ONNX model into your browser

You can run the app directly hosted on Glitch here: https://gabehollombe-aws-sagemaker-image-classifier-to-onnx-in-browser.glitch.me

1. Select your ONNX model and paste your list of class labels into the webpage
2. Capture some webcam images, drag some from your filesystem, or paste URLs and watch the inference happen! (The first one takes a few moments to load the model, and subsequent images will infer pretty quickly)

Or remix it yourself! 

<!-- Remix Button -->
<a href="https://glitch.com/edit/#!/remix/gabehollombe-aws-sagemaker-image-classifier-to-onnx-in-browser">
  <img src="https://cdn.glitch.com/2bdfb3f8-05ef-4035-a06e-2043962a3a13%2Fremix%402x.png?1513093958726" alt="remix this" height="33">
</a>

Or, run the frontend single-page-app yourself locally. Just run `npm install && npm start` to install the web app dependencies and start the single-page-app in your browser.
