import { StatusBar } from 'expo-status-bar';
import React,{useState,useEffect} from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Camera } from "expo-camera";
import { cameraWithTensors, bundleResourceIO } from "@tensorflow/tfjs-react-native";
import * as tf from "@tensorflow/tfjs";
const modelJson = require('./assets/models/model.json');
const modelWeights = require("./assets/models/weights.bin");
const TensorCamera = cameraWithTensors(Camera);

export default function App() {
  const [isModelRead, setIsModelRead] = useState(false);
  const [useModel, setUseModel] = useState({});
  const [model, setModel] = useState(null);
  const [cameraPermission, setCameraPermission] = useState(false)
  const [predictions, setPredictions] = useState([])

  useEffect( () => {
    (async ()=>{
      await tf.ready();
      const {status } = await Camera.requestCameraPermissionsAsync()
      console.log(status)
      setCameraPermission(status == "granted")
      const newmodel = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
      setIsModelRead(true),
      setModel(newmodel)
      console.log("model loaded")
      console.log(cameraPermission)
    })()
    
  }, [])

  let textureDims;
  if (Platform.OS === "ios") {
    textureDims = {
      height: 1920,
      width: 1080,
    };
  } else {
    textureDims = {
      height: 1200,
      width: 1600,
    };
  }

  const makeHandleCameraStream = (imageAsTensors)=> {
    if (!imageAsTensors) {
      console.log("Image not found!");
    }
      const loop = async () => {
          const  imageTensor = imageAsTensors.next().value;
          try {
            const alignCorners = true;
            // Resize from (320,240,3) -> (224,224,3)
            const imageResize = tf.image.resizeBilinear(
              imageTensor,
              [224, 224],
              alignCorners
            );
            let expandedImageTensor = tf.expandDims(imageResize,0)
            
            const predictions = await model.predict(expandedImageTensor);
            // console.log(predictions)
            setPredictions(predictions)
            
          } catch (error) {
            console.log(error.message)
          }
          tf.dispose(imageTensor);
      };
      loop();
  }


  return (
    <View>
        {model && (
          <TensorCamera
            // Standard Camera props
            style={styles.camera}
            type={Camera.Constants.Type.front}
            // Tensor related props
            cameraTextureHeight={textureDims.height}
            cameraTextureWidth={textureDims.width}
            resizeHeight={320}
            resizeWidth={240}
            resizeDepth={3}
            onReady={makeHandleCameraStream()}
            autorender={true}
          />
        )}

        {/* {predictions && 
            <View>
            {predictions.map(p => <Text>{p.className}</Text>)}
            </View>
        } */}

      </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    height: 320,
    width: 240,
  },
});
