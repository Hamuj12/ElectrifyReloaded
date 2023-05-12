import React, { useState } from "react";
import { StyleSheet, View, Image, ActivityIndicator, FlatList, SafeAreaView, ScrollView } from "react-native";
import { Button } from "react-native-elements";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";

export default function App() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [croppedImages, setCroppedImages] = useState([]);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setCroppedImages([]);
    }
  };

  const processImage = async () => {
    setLoading(true);
    const base64 = await uriToBase64(image);
    axios
      .post("http://192.168.68.62:5000/predict", { image: base64 })
      .then((response) => {
        const base64Result = response.data.image;
        const croppedImages = response.data.cropped_images;
        console.log('Number of cropped images:', croppedImages.length);  // New console log
        setCroppedImages(croppedImages);
        setImage(`data:image/jpeg;base64,${base64Result}`);
      })
      .catch((error) => {
        console.error(error);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const uriToBase64 = async (uri) => {
    const response = await fetch(uri);
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.buttonContainer}>
        <Button title="Pick an image" onPress={pickImage} />
        <Button title="Process image" onPress={processImage} disabled={!image} />
      </View>
      {loading ? (
        <ActivityIndicator size="large" color="#0000ff" />
      ) : (
        <View style={styles.imagesContainer}>
          <Image source={{ uri: image }} style={styles.image} resizeMode="contain" />
          <View style={styles.listContainer}>
            <FlatList
              data={croppedImages}
              renderItem={({ item }) => (
                <Image
                  source={{ uri: `data:image/jpeg;base64,${item}` }}
                  style={styles.croppedImage}
                  resizeMode="contain"
                />
              )}
              keyExtractor={(item, index) => index.toString()}
            />

          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
    marginBottom: 10,
  },
  imagesContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: '100%',
    height: '40%',
  },
  listContainer: {
    flex: 1,
    width: '100%',
  },
  croppedImage: {
    width: '100%',
    height: 200,  // or any other height you prefer
  },
});
