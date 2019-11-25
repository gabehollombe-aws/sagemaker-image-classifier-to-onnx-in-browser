import React from 'react';
import { Form } from 'semantic-ui-react'
import Webcam from 'react-webcam';

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
    width: props.imageWidth,
    height: props.imageHeight,
    facingMode: "user"
  };

  return (
    <div>
      <div>
        <Webcam
          audio={false}
          height={props.width}
          width={props.height}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
        />
      </div>

      <Form.Button onClick={capture}>Classify Webcam Image</Form.Button>
    </div>
  );
}

export default WebcamCapture