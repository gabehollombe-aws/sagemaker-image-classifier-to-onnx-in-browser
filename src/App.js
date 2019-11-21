/* global ndarray */

import React, { Component } from 'react';
import { Accordion, Button, Card, CardGroup, Form, Image } from 'semantic-ui-react'
import Webcam from 'react-webcam';
import classNames from 'classnames'
import Dropzone from 'react-dropzone'
import { Tensor, InferenceSession } from 'onnxjs'

const IMAGE_WIDTH = 224;
const IMAGE_HEIGHT = 224;

const styles = {
  dropZone: {
    width: '100%',
    height: '200px',
    border: '1px solid gray',
    margin: '10px 0',
  }
}

// argMax via https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function zipArrays(a, b) {
  return a.map((e, i) => [e, b[i]])
}

// scaleImage via https://gist.github.com/MikeRogers0/6264546
function scaleImage(url, width, height, callback){
	let img = new window.Image();

	img.onload = function(){
		var canvas = document.createElement("canvas"),
        ctx = canvas.getContext("2d");

        canvas.width = width;
        canvas.height = height;

        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // resize code via https://sdqali.in/blog/2013/10/03/fitting-an-image-in-to-a-canvas-object/
        var imageAspectRatio = img.width / img.height;
        var canvasAspectRatio = canvas.width / canvas.height;
        var renderableHeight, renderableWidth, xStart, yStart;
        if(imageAspectRatio < canvasAspectRatio) {
          renderableHeight = canvas.height;
          renderableWidth = img.width * (renderableHeight / img.height);
          xStart = (canvas.width - renderableWidth) / 2;
          yStart = 0;
        }
        else if(imageAspectRatio > canvasAspectRatio) {
          renderableWidth = canvas.width
          renderableHeight = img.height * (renderableWidth / img.width);
          xStart = 0;
          yStart = (canvas.height - renderableHeight) / 2;
        }
        else {
          renderableHeight = canvas.height;
          renderableWidth = canvas.width;
          xStart = 0;
          yStart = 0;
        }
        ctx.drawImage(img, xStart, yStart, renderableWidth, renderableHeight);

        callback(canvas);
	};

  img.src = url;
}


class WebcamCapture extends React.Component {
  setRef = webcam => {
    this.webcam = webcam;
  };

  handleCapture=() => {
    this.props.onCapture(this.webcam.getScreenshot())
  }

  render() {
    const videoConstraints = {
      width: IMAGE_WIDTH,
      height: IMAGE_HEIGHT,
      facingMode: "user"
    };

    return (
      <div>
        <div>
          <Webcam
            audio={false}
            height={IMAGE_HEIGHT}
            width={IMAGE_WIDTH}
            ref={this.setRef}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
          />
        </div>

        <Form.Button onClick={this.handleCapture}>Classify</Form.Button>
      </div>
    );
  }
}

class ClassifiedImage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      classLabel: null,
      probability: null,
    }
  }

  async componentDidMount() {
    const { classLabels, predictions, highestProbabilityIndex } = await this.props.classifier.classify(this.props.imageData);
    let sortedClassLabels = classLabels.splice(0)
    sortedClassLabels.sort()
    this.setState({ 
      bestLabel: sortedClassLabels[highestProbabilityIndex],
      bestLabelScore: predictions[highestProbabilityIndex],
      allLabelsScores: zipArrays(sortedClassLabels, predictions),
    })
  }

  accordionPanels = () => {
    if (!this.state.allLabelsScores) return [];

    const labelsAndScores = this.state.allLabelsScores.map(([label, score]) => 
      <p>{label}: {score}</p>
    )
    return [{
      key: 'labels-and-scores',
      title: 'Show Score Details',
      content: labelsAndScores
    }]
  }

  render() {
    return (
      <Card style={{width: '224px'}}>
        <Image src={this.props.imageDataUrl} />

        <Card.Content>
          <Card.Header>
            { this.state.bestLabel ? this.state.bestLabel : "Loading..." }
          </Card.Header>

          <Card.Meta>
            { this.state.bestLabelScore ? this.state.bestLabelScore : "" }
          </Card.Meta>

          <Card.Description>
            <Accordion defaultActiveIndex={-1} panels={this.accordionPanels()} />
          </Card.Description>
        </Card.Content>
      </Card>
    )
  }
}



class Classifier {
  constructor(modelData, classLabels, imageWidth, imageHeight) {
    this.imageWidth = imageWidth
    this.imageHeight = imageHeight
    this.classLabels = classLabels
    this.modelData = modelData
   
    this.onnxSession = new InferenceSession()
  }

  loadModel = async () => {
    console.log('loading ', this.modelData)
    await this.onnxSession.loadModel(this.modelData)
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

  async classify(imageData) {
    const preprocessedData = this.realignImageDataForInference(imageData.data, this.imageWidth, this.imageHeight)
    const inputTensor = new Tensor(preprocessedData, 'float32', [1, 3, IMAGE_WIDTH, IMAGE_HEIGHT]);
    const outputMap = await this.onnxSession.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;
    const predictions = Array.from(outputData);
    const highestProbabilityIndex = argMax(predictions);
    const classLabels = [].concat(this.classLabels.split(' '));
    classLabels.sort();

    return {
      classLabels, predictions, highestProbabilityIndex
    }
  }
}



class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      images: [],
      onnxModel: null,
      classLabels: '',
      loadingModel: false,
    }
    this.fileInputRef = React.createRef()
    this.classifier = null
  }

  classify = async (imageDataUrl, imageData) => {
    this.state.classifier.setLabels(this.state.classLabels)
    this.setState({
      images: [...this.state.images, { imageDataUrl, imageData }]
    })
  };

  handleChange = (e) => this.setState({ [e.target.name]: e.target.value })

  classifyScaled = (canvas) => {
    const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height)
    const imageDataUrl = canvas.toDataURL()
    this.classify(imageDataUrl, imageData)
  }

  onDrop = (acceptedFiles, rejectedFiles) => {
    // Do something with files
    acceptedFiles.forEach(f => {
      var reader  = new FileReader();
      reader.addEventListener("load", () => {
        scaleImage(reader.result, IMAGE_WIDTH, IMAGE_HEIGHT, this.classifyScaled)
      }, false);
      reader.readAsDataURL(f);
    })
  }

  handleClearImages = () => {
    this.setState({
      images: []
    })
  }

  updateClassifier = async (modelData) => {
    const classifier = new Classifier(modelData, this.state.classLabels, IMAGE_WIDTH, IMAGE_HEIGHT)
    await classifier.loadModel()
    this.setState({ classifier })
    console.log('Classifier updated.')
  }

  handleModelChanged = async (e) => {
    const fileName = e.target.value
    const data = e.target.files[0]
    await this.updateClassifier(data)
  }

  render() {
    return (
      <div>
      <Form>
        <Form.Group widths='equal'>
          <Button
            content="Select ONNX Model"
            labelPosition="left"
            icon="file"
            onClick={() => this.fileInputRef.current.click()}
          />

          <input
            ref={this.fileInputRef}
            type="file"
            hidden
            onChange={this.handleModelChanged}
          />

          <Form.Input label='Class Labels' placeholder='space delimited list of labels' name='classLabels' onChange={this.handleChange} value={this.state.classLabels} />
        </Form.Group>

        <Form.Group widths='equal'>
          <WebcamCapture onCapture={this.classify}/>
        </Form.Group>
      </Form>

      <Dropzone onDrop={this.onDrop} accept={['image/jpg', 'image/jpeg', 'image/png']}>
        {({getRootProps, getInputProps, isDragActive}) => {
          return (
            <div 
              {...getRootProps()}
              style={styles.dropZone}
              className={classNames('dropzone', {'dropzone--isActive': isDragActive})}
            >
              <input {...getInputProps()} />
              {
                isDragActive ?
                  <p>Drop files here...</p> :
                  <p>You can drag and drop images here or click to select images to upload.</p>
              }
            </div>
          )
        }}
      </Dropzone>

      <Button onClick={this.handleClearImages}>Clear Images</Button>

      <CardGroup>
        { this.state.images.map(({imageDataUrl, imageData}, index) => <ClassifiedImage key={"img"+index} imageDataUrl={imageDataUrl} imageData={imageData} classifier={this.state.classifier} />) }
      </CardGroup>
      </div>
    );
  }
}

export default App;
