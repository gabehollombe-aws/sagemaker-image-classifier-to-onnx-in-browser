/* global ndarray */
import { Tensor, InferenceSession } from 'onnxjs'
import { argMax } from '../utils'
import axios from 'axios'

class Classifier {
  constructor(imageWidth, imageHeight) {
    this.imageWidth = imageWidth
    this.imageHeight = imageHeight
    this.onnxSession = new InferenceSession()
    // this.sagemakerEndpointUrl = '',
    this.canvas = document.createElement('canvas')
    this.canvas.width = imageWidth
    this.canvas.height = imageHeight
    this.ctx = this.canvas.getContext('2d')
  }

  loadModel = async (modelData) => {
    await this.onnxSession.loadModel(modelData)
  }

  setLabels = (classLabels) => {
    this.classLabels = classLabels
  }

  realignImageDataForInference(data, width, height) {
    // Preprocess raw image data to match SageMaker's image classifier expected shape
    // re-aligning the imageData from [224*224*4] to the correct dimension [1*3*224*224]
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);
    ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
    ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));

    return dataProcessed.data;
  }

  getImageData = (src) => {
    return new Promise((resolve, reject) => {
      const img = new window.Image()
      img.onload = () => {
        this.ctx.drawImage(img, 0, 0)
        img.style.display = 'none'
        resolve(this.ctx.getImageData(0, 0, this.imageWidth, this.imageHeight))
      }
      img.onerror = reject
      img.src = src
    })
  }

  async classify(imageSrc) {
    const imageData = await this.getImageData(imageSrc)
    const preprocessedData = this.realignImageDataForInference(imageData.data, this.imageWidth, this.imageHeight)
    const inputTensor = new Tensor(preprocessedData, 'float32', [1, 3, this.imageWidth, this.imageHeight]);
    const outputMap = await this.onnxSession.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;
    return this.makeResponse(Array.from(outputData))
  }

  async classifyHosted(dataUri) {
    const result = await axios.post(
        this.sagemakerEndpointUrl, 
        dataURItoBlob(dataUri), 
        { headers: { 'Content-Type': 'application/x-image'}})
    return this.makeResponse(result.data)
  }

  makeResponse(predictions) {
    const highestProbabilityIndex = argMax(predictions);
    const classLabels = [].concat(this.classLabels.split(' '));
    classLabels.sort();
    return {
      classLabels, predictions, highestProbabilityIndex
    }
  }
}

function dataURItoBlob(dataURI) {
    // convert base64 to raw binary data held in a string via
    // https://stackoverflow.com/a/7261048/30632
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0]
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], {type: mimeString});
}

export default Classifier