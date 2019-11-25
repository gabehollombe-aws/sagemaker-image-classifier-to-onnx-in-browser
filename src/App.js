import React, { Component } from 'react';
import { Button, CardGroup, Container, Divider, Form, Header, Icon, Label, Segment } from 'semantic-ui-react'
import Dropzone from 'react-dropzone'

import { scaleImage } from './utils'

import WebcamCapture from './components/WebcamCapture'
import ClassifiedImage from './components/ClassifiedImage'
import Classifier from './components/Classifier'

const IMAGE_WIDTH = 224;
const IMAGE_HEIGHT = 224;

class App extends Component {
  constructor(props) {
    super(props);
    const classifier = new Classifier(IMAGE_WIDTH, IMAGE_HEIGHT)
    this.state = {
      images: [],
      onnxModel: null,
      classLabels: '',
      // classLabels: 'BACKGROUND_Google Faces Faces_easy Leopards Motorbikes accordion airplanes anchor ant barrel bass beaver binocular bonsai brain brontosaurus buddha butterfly camera cannon car_side ceiling_fan cellphone chair chandelier cougar_body cougar_face crab crayfish crocodile crocodile_head cup dalmatian dollar_bill dolphin dragonfly electric_guitar elephant emu euphonium ewer ferry flamingo flamingo_head garfield gerenuk gramophone grand_piano hawksbill headphone hedgehog helicopter ibis inline_skate joshua_tree kangaroo ketch lamp laptop llama lobster lotus mandolin mayfly menorah metronome minaret nautilus octopus okapi pagoda panda pigeon pizza platypus pyramid revolver rhino rooster saxophone schooner scissors scorpion sea_horse snoopy soccer_ball stapler starfish stegosaurus stop_sign strawberry sunflower tick trilobite umbrella watch water_lilly wheelchair wild_cat windsor_chair wrench yin_yang',
      addImageFromUrl: '',
      selectedModelFileName: '',
      loadingModel: false,
      modelLoadError: '',
      classifier,
      modelLocation: 'local',
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

  handleChange = (e, {name, value}) => {
    this.setState({ [name]: value })
  }

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

  setClassifierModelData = async (modelData) => {
    try {
      this.setState({ modelLoadError: '' })
      await this.state.classifier.loadModel(modelData)
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

    await this.setClassifierModelData(file)
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
              content={ this.state.selectedModelFileName === '' ? 'Click to select your ONNX model file from your computer' : this.state.selectedModelFileName }
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
            { this.state.classLabels.length === 0 &&
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

          <Divider horizontal>
            <Header as='h3'>
              <Icon name='video camera' />
              Classify via Webcam
            </Header>
          </Divider>
          <WebcamCapture onCapture={this.classify} width={IMAGE_WIDTH} height={IMAGE_HEIGHT} />

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
                if (e.keyCode === 13) {
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
          {({getRootProps, getInputProps}) => {
            return (
              <Segment secondary>
                <div {...getRootProps()}>
                  <input {...getInputProps()} />
                  <Header as='h4'>Drag and drop images here or click to select images to classify.</Header>
                  <Icon name='images outline' size='massive' />
                </div>
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
