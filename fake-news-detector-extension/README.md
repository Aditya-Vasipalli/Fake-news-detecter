# Fake News Detector Chrome Extension

This project is a Chrome extension designed to detect fake news articles using a machine learning model. It provides users with an easy-to-use interface to analyze news content and receive predictions on its authenticity.

## Project Structure

```
fake-news-detector-extension
├── src
│   ├── background.js        # Background script managing events and interactions
│   ├── content.js          # Content script for DOM manipulation on web pages
│   ├── popup.js            # Script for the popup interface
│   ├── api
│   │   └── index.js        # API functions for interacting with the fake news detection model
│   └── styles
│       └── popup.css       # CSS styles for the popup interface
├── public
│   ├── popup.html          # HTML structure for the popup interface
│   └── icon.png            # Icon for the Chrome extension
├── manifest.json           # Configuration file for the Chrome extension
└── README.md               # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd fake-news-detector-extension
   ```

3. Open Chrome and go to `chrome://extensions/`.

4. Enable "Developer mode" in the top right corner.

5. Click on "Load unpacked" and select the `fake-news-detector-extension` directory.

## Usage

1. Click on the extension icon in the Chrome toolbar to open the popup interface.

2. Enter the text or URL of the news article you want to analyze.

3. Click the "Analyze" button to receive a prediction on whether the article is real or fake.

4. Review the results displayed in the popup.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.