document.addEventListener('DOMContentLoaded', function () {
  const analyzeButton = document.getElementById('analyze-button');
  const resultDiv = document.getElementById('result');
  const spinner = document.getElementById('spinner');

  let currentTabUrl = null;

  // Get the current tab's URL
  if (chrome && chrome.tabs) {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      if (tabs && tabs.length > 0) {
        currentTabUrl = tabs[0].url;
      }
    });
  }

  analyzeButton.addEventListener('click', async () => {
    analyzeButton.classList.add('loading');
    const url = currentTabUrl;
    if (!url) {
      resultDiv.textContent = 'Could not get the current tab URL.';
      return;
    }

    spinner.style.display = 'block'; // Show spinner
    resultDiv.textContent = 'Analyzing...';

    try {
      const response = await fetch(`http://127.0.0.1:8000/predict?url=${encodeURIComponent(url)}`);
      if (!response.ok) {
        resultDiv.textContent = 'Error: Could not analyze the article.';
        return;
      }
      const data = await response.json();

      // Build SHAP keywords HTML
      let shapHtml = '';
      if (data.top_keywords && data.top_keywords.length > 0) {
        shapHtml = '<b>Top SHAP Keywords:</b><ul style="padding-left:18px">';
        data.top_keywords.forEach(kw => {
          shapHtml += `<li>${kw.word}: <b>${kw.shap_value.toFixed(2)}</b> (${kw.direction})</li>`;
        });
        shapHtml += '</ul>';
      }

      // Optionally, show the SHAP plot image
      let plotHtml = `<img src="http://127.0.0.1:8000/plot?${Date.now()}" alt="SHAP Plot" style="max-width:100%;margin-top:8px"/>`;

      resultDiv.innerHTML = `
        <b>${data.title}</b><br>
        Prediction: <b>${data.prediction}</b><br>
        Confidence: Real ${data.confidence_real}, Fake ${data.confidence_fake}<br>
        ${shapHtml}
        ${plotHtml}
      `;

      // Send keywords to content script for highlighting
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { action: "highlight_keywords", keywords: data.top_keywords }
        );
      });
    } catch (error) {
      resultDiv.textContent = 'Error: Could not connect to backend.';
      console.error(error);
    } finally {
      spinner.style.display = 'none'; // Hide spinner after fetch
      analyzeButton.classList.remove('loading');
    }
  });

  const toggleTextInput = document.getElementById('toggle-text-input');
  const textInput = document.getElementById('text-input');

  toggleTextInput.addEventListener('click', function (e) {
    e.preventDefault();
    if (textInput.style.display === 'block') {
      textInput.style.display = 'none';
      toggleTextInput.textContent = 'Or paste text instead';
    } else {
      textInput.style.display = 'block';
      toggleTextInput.textContent = 'Hide text input';
    }
  });
});

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "highlight_keywords") {
    highlightKeywords(request.keywords);
  }
});

function highlightKeywords(keywords) {
  if (!keywords || keywords.length === 0) return;

  // Build a regex for all keywords (case-insensitive)
  const words = keywords.map(k => k.word).filter(Boolean);
  if (words.length === 0) return;
  const regex = new RegExp(`\\b(${words.join('|')})\\b`, 'gi');

  // Walk the DOM and replace text nodes
  function walk(node) {
    if (node.nodeType === 3) { // Text node
      const parent = node.parentNode;
      if (!parent) return;
      const frag = document.createDocumentFragment();
      let lastIdx = 0;
      let match;
      let text = node.nodeValue;
      regex.lastIndex = 0;
      while ((match = regex.exec(text)) !== null) {
        const before = text.slice(lastIdx, match.index);
        if (before) frag.appendChild(document.createTextNode(before));
        const word = match[0];
        const kw = keywords.find(k => k.word.toLowerCase() === word.toLowerCase());
        if (kw) {
          const span = document.createElement('span');
          // Color and intensity
          const intensity = Math.min(Math.abs(kw.shap_value) / 2, 1); // scale for effect
          if (kw.direction.includes('Fake')) {
            span.style.background = `rgba(255,0,0,${0.2 + 0.6 * intensity})`;
          } else {
            span.style.background = `rgba(0,200,0,${0.2 + 0.6 * intensity})`;
          }
          span.style.borderRadius = '3px';
          span.style.padding = '0 2px';
          span.style.transition = 'background 0.3s';
          span.title = `SHAP: ${kw.shap_value.toFixed(2)} (${kw.direction})`;
          span.textContent = word;
          frag.appendChild(span);
        } else {
          frag.appendChild(document.createTextNode(word));
        }
        lastIdx = regex.lastIndex;
      }
      const after = text.slice(lastIdx);
      if (after) frag.appendChild(document.createTextNode(after));
      parent.replaceChild(frag, node);
    } else if (node.nodeType === 1 && node.childNodes && !['SCRIPT','STYLE','NOSCRIPT','IFRAME'].includes(node.tagName)) {
      for (let i = 0; i < node.childNodes.length; i++) {
        walk(node.childNodes[i]);
      }
    }
  }

  walk(document.body);
}