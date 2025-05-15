// This is the background script for the Chrome extension.
// It manages events and handles interactions between different parts of the extension.

chrome.runtime.onInstalled.addListener(() => {
    console.log("Fake News Detector Extension installed.");
});

// Listen for messages from content scripts or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzeText") {
        // Handle text analysis request
        analyzeText(request.text).then(response => {
            sendResponse({ result: response });
        });
        return true; // Indicates that the response will be sent asynchronously
    }
});

// Function to analyze text using the fake news detection API
async function analyzeText(text) {
    const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ content: text })
    });
    const data = await response.json();
    return data;
}