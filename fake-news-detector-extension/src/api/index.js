import axios from 'axios';

const API_URL = 'http://localhost:5000/predict'; // Adjust the URL as needed

export const analyzeText = async (text) => {
    try {
        const response = await axios.post(API_URL, { content: text });
        return response.data; // Assuming the response contains the prediction
    } catch (error) {
        console.error('Error analyzing text:', error);
        throw error; // Rethrow the error for handling in the calling function
    }
};