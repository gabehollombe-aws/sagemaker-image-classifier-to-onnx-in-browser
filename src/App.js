/* global ndarray */

import React, { Component } from 'react';
import { Accordion, Button, Card, CardGroup, Container, Dimmer, Divider, Form, Header, Icon, Image, Label, Loader, Segment } from 'semantic-ui-react'
import Webcam from 'react-webcam';
import classNames from 'classnames'
import Dropzone from 'react-dropzone'
import { Tensor, InferenceSession } from 'onnxjs'
import { v4 as uuid } from 'uuid'

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


const WebcamCapture = (props) => {
  const webcamRef = React.useRef(null); 

  const capture = React.useCallback(
    () => {
      const imageSrc = webcamRef.current.getScreenshot();
      props.onCapture(imageSrc)
    },
    [webcamRef]
  );

  const videoConstraints = {
    width: IMAGE_WIDTH,
    height: IMAGE_HEIGHT,
    facingMode: "user"
  };

  return (
    <div>
      <Divider horizontal>
        <Header as='h3'>
          <Icon name='video camera' />
          Classify via Webcam
        </Header>
      </Divider>
      <div>
        <Webcam
          audio={false}
          height={IMAGE_HEIGHT}
          width={IMAGE_WIDTH}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
        />
      </div>

      <Form.Button onClick={capture}>Classify Webcam Image</Form.Button>
    </div>
  );
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
    const { classLabels, predictions, highestProbabilityIndex } = await this.props.classifier.classify(this.props.imageDataUrl);
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
      <p key={uuid()}>{label}: {score}</p>
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
    this.canvas = document.createElement('canvas')
    this.canvas.width = IMAGE_WIDTH
    this.canvas.height = IMAGE_HEIGHT
    this.ctx = this.canvas.getContext('2d')
  }

  loadModel = async () => {
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

  getImageData = (src) => {
    return new Promise((resolve, reject) => {
      const img = new window.Image()
      img.onload = () => {
        this.ctx.drawImage(img, 0, 0)
        img.style.display = 'none'
        resolve(this.ctx.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT))
      }
      img.onerror = reject
      img.src = src
    })
  }

  async classify(imageSrc) {
    const imageData = await this.getImageData(imageSrc)
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
      // classLabels: 'BACKGROUND_Google Faces Faces_easy Leopards Motorbikes accordion airplanes anchor ant barrel bass beaver binocular bonsai brain brontosaurus buddha butterfly camera cannon car_side ceiling_fan cellphone chair chandelier cougar_body cougar_face crab crayfish crocodile crocodile_head cup dalmatian dollar_bill dolphin dragonfly electric_guitar elephant emu euphonium ewer ferry flamingo flamingo_head garfield gerenuk gramophone grand_piano hawksbill headphone hedgehog helicopter ibis inline_skate joshua_tree kangaroo ketch lamp laptop llama lobster lotus mandolin mayfly menorah metronome minaret nautilus octopus okapi pagoda panda pigeon pizza platypus pyramid revolver rhino rooster saxophone schooner scissors scorpion sea_horse snoopy soccer_ball stapler starfish stegosaurus stop_sign strawberry sunflower tick trilobite umbrella watch water_lilly wheelchair wild_cat windsor_chair wrench yin_yang',
      addImageFromUrl: '',
      selectedModelFileName: '',
      loadingModel: false,
      modelLoadError: '',
    }
    this.fileInputRef = React.createRef()
    this.classifier = null
  }

  classify = async (imageDataUrl) => {
    this.state.classifier.setLabels(this.state.classLabels)
    this.setState({
      images: [...this.state.images, { imageDataUrl}]
    })
  };

  handleChange = (e) => this.setState({ [e.target.name]: e.target.value })

  classifyScaled = (canvas) => {
    const imageDataUrl = canvas.toDataURL()
    this.classify(imageDataUrl)
  }

  onDrop = (acceptedFiles) => {
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

  addImageFromUrl = () => {
    this.classify(this.state.addImageFromUrl)
    this.setState({
      addImageFromUrl: ''
    })
  }

  updateClassifier = async (modelData) => {
    const classifier = new Classifier(modelData, this.state.classLabels, IMAGE_WIDTH, IMAGE_HEIGHT)

    try {
      this.setState({ modelLoadError: '' })
      await classifier.loadModel()
      this.setState({ classifier })
    }
    catch(ex) {
      console.error(ex)
      this.setState({ modelLoadError: ex.message })
    }

  }

  handleModelChanged = async (e) => {
    const file = e.target.files[0]
    this.setState({
      selectedModelFileName: file.name,
      loadingModel: true,
    })

    await this.updateClassifier(file)
    this.setState({ loadingModel: false })
  }

  render() {
    return (
      <Container>

      <Header as='h1'>Custom image classification at the edge, in your web browser!</Header>

      <p>
        This page makes it easy to try out a custome image classifer trained via Amazon SageMaker and exported as ONNX format.
      </p>

      <p>
        For an example Jupyter notebook that shows you how to train your own custom image classifier model with Amazon Sagemker, see <br/>
        <a hreef="https://github.com/gabehollombe-aws/sagemaker-image-classifier-to-onnx-in-browser/blob/master/sagemaker/train_and_export_as_onnx.ipynb">https://github.com/gabehollombe-aws/sagemaker-image-classifier-to-onnx-in-browser/blob/master/sagemaker/train_and_export_as_onnx.ipynb</a>
      </p>

      <Segment>
        <Header as='h2'>1. Specify your model and class labels</Header>

        <Form onSubmit={(e)=>e.preventDefault()}>
          <Form.Group widths='equal'>
            <Button
              content={ this.state.selectedModelFileName == '' ? 'Click to select your ONNX model file from your computer' : this.state.selectedModelFileName }
              labelPosition="left"
              icon="file"
              onClick={() => this.fileInputRef.current.click()}
              color={ this.state.selectedModelFileName ? null : 'red' }
              loading={this.state.loadingModel}
            />

            <input
              ref={this.fileInputRef}
              type="file"
              hidden
              onChange={this.handleModelChanged}
            />

            { this.state.modelLoadError &&
              <Label as='h3' color='red'>
                <Icon name='exclamation triangle' size='large'/>
                Error loading model: { this.state.modelLoadError } <br/> <br/> Are you sure the selected file is an ONNX formatted model?
              </Label>
            }

          </Form.Group>


          <Form.Group widths='equal'>
            <Form.Field>

             <Form.Input label='Class Labels' placeholder='Paste a space delimited list of your class labels here' name='classLabels' onChange={this.handleChange} value={this.state.classLabels} />
            { this.state.classLabels.length == 0 &&
              <Label pointing color='red'>You must enter a space-delimited list of class labels that your model will score on. For exampe: 'dog cat human'</Label>
            }
            </Form.Field>
          </Form.Group>
        </Form>
      </Segment>

      { 
        this.state.selectedModelFileName && this.state.classLabels &&
      <Segment>
        <Header as='h2'>2. Add some images</Header>
          <WebcamCapture onCapture={this.classify}/>

          <Divider section horizontal>
          <Header as='h3'>
            <Icon name='globe' />
            Classify via Image URLS
          </Header>
        </Divider>

        <Form>
          <Form.Group widths='equal'>
              <Form.Input 
              placeholder='http://path/to/some/image.jpg' 
              name='addImageFromUrl' 
              onChange={this.handleChange} 
              value={this.state.addImageFromUrl} 
              action={{
                content: 'Classify Image From URL',
                onClick: () => this.addImageFromUrl()
              }}
              onKeyDown={(e) => {
                if (e.keyCode == 13) {
                  e.preventDefault()
                  this.addImageFromUrl()
                }
              }}
              />
          </Form.Group>
        </Form>

        <Divider section horizontal>
          <Header as='h3'>
            <Icon name='file image' />
            Classify via image files
          </Header>
        </Divider>

        <Dropzone onDrop={this.onDrop} accept={['image/jpg', 'image/jpeg', 'image/png']}>
          {({getRootProps, getInputProps, isDragActive}) => {
            return (
              <Segment secondary
                {...getRootProps()}
                style={styles.dropZone}
                className={classNames('dropzone', {'dropzone--isActive': isDragActive})}
              >
                <input {...getInputProps()} />
                <Header as='h4'>Drag and drop images here or click to select images to classify.</Header>
                <Icon name='images outline' size='massive' />
              </Segment>
            )
          }}
        </Dropzone>
      </Segment>
      }

      { this.state.images.length > 0 &&
        <Segment>
        <Header as='h2'>3. View the results</Header>

        <Segment basic>
          <Button onClick={this.handleClearImages}>Clear Images</Button>
        </Segment>

        <CardGroup>
          { this.state.images.map(({imageDataUrl, imageData}, index) => <ClassifiedImage key={"img"+index} imageDataUrl={imageDataUrl} classifier={this.state.classifier} />) }
        </CardGroup>
      </Segment>
      }

    </Container>
    );
  }
}

export default App;
