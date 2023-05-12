import React, { useState } from "react";
import { StyleSheet, View, Image, ActivityIndicator } from "react-native";
import { Button } from "react-native-elements";
import * as ImagePicker from "expo-image-picker";
import axios from "axios";

export default function App() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const processImage = async () => {
    setLoading(true);
    const base64 = await uriToBase64(image);
    axios
      .post("http://192.168.68.62:5000/predict", { image: base64 })
      .then((response) => {
        const base64Result = response.data.image;
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
    <View style={styles.container}>
      <Button title="Pick an image" onPress={pickImage} />
      <Button title="Process image" onPress={processImage} disabled={!image} />
      {loading ? (
        <ActivityIndicator size="large" color="#0000ff" />
      ) : (
        image && (
          <Image source={{ uri: image }} style={{ width: "100%", height: "80%" }} resizeMode="contain" />
        )
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});

