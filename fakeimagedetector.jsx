import { useState } from "react";

export default function FakeImageDetector() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }
    setLoading(true);
    setResult(null);
    setError(null);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Failed to process image. Please try again.");
      }
      const data = await response.json();
      setResult(data.prediction);
    } catch (error) {
      console.error("Error detecting image:", error);
      setError(error.message || "An unexpected error occurred.");
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
      <nav className="w-full bg-white shadow-md p-4 flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-700">Fake Image Detector</h1>
        <div className="flex space-x-6">
          <a href="#" className="text-gray-700 font-semibold hover:text-blue-500">Home</a>
          <a href="#" className="text-gray-700 font-semibold hover:text-blue-500">FAQs</a>
          <a href="#" className="text-gray-700 font-semibold hover:text-blue-500">Blog</a>
          <a href="#" className="text-gray-700 font-semibold hover:text-blue-500">About Us</a>
          <a href="#" className="text-gray-700 font-semibold hover:text-blue-500">Contact Us</a>
        </div>
      </nav>
      <div className="bg-white shadow-lg rounded-2xl p-6 w-96 text-center">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleImageUpload} 
          className="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600"
        />
        {preview && (
          <img 
            src={preview} 
            alt="Uploaded Preview" 
            className="w-full h-48 object-cover rounded-lg mb-4 border"
          />
        )}
        <button 
          onClick={handleSubmit} 
          className={`w-full px-4 py-2 rounded-lg text-white font-semibold ${loading ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-600'}`} 
          disabled={loading}
        >
          {loading ? "Processing..." : "Upload & Detect"}
        </button>
        {loading && (
          <div className="mt-4 flex justify-center">
            <div className="w-6 h-6 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        )}
        {error && (
          <p className="mt-4 text-lg font-semibold text-red-600">Error: {error}</p>
        )}
        {result && (
          <p className="mt-4 text-lg font-semibold text-gray-700">Result: <span className="text-blue-600">{result}</span></p>
        )}
      </div>
    </div>
  );
}
