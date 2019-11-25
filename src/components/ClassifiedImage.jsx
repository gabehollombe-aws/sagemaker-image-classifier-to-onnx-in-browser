import React, { Component } from 'react';
import { Accordion, Card, Image } from 'semantic-ui-react'
import { v4 as uuid } from 'uuid'
import { zipArrays } from '../utils'

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

export default ClassifiedImage