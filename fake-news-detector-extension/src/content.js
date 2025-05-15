// This file contains the content script for the Chrome extension.
// It interacts with the web page to highlight keywords related to fake news articles.

// Listen for messages from popup.js to highlight keywords
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