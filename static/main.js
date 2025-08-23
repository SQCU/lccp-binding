document.addEventListener("DOMContentLoaded", () => {
    // --- 1. DEFINE ALL DOM ELEMENTS ---
    // Tabs & Panes
    const tabBtnMultiStream = document.getElementById('tab-btn-multi-stream');
    const tabBtnNgram = document.getElementById('tab-btn-ngram');
    const paneMultiStream = document.getElementById('pane-multi-stream');
    const paneNgram = document.getElementById('pane-ngram');
    // Multi-Stream Pane
    const mainPrompt = document.getElementById("main-prompt");
    const halfCtxBtn = document.getElementById("half-ctx-btn");
    const generateBtn = document.getElementById("generate-btn");
    const promptsContainer = document.getElementById("prompts-container");
    const streamsContainer = document.getElementById("streams-container");
    // Context Slice Pane
    const ngramInput = document.getElementById('ngram-input');
    const ngramProbeBtn = document.getElementById('ngram-probe-btn');
    const ngramResultsContainer = document.getElementById('ngram-results-container');
    // ⭐ NEW: Filter controls
    const filterControls = document.getElementById('filter-controls');
    const topKSlider = document.getElementById('top-k-slider');
    const topPSlider = document.getElementById('top-p-slider');
    const minPSlider = document.getElementById('min-p-slider');
    const topKLabel = document.getElementById('top-k-label');
    const topPLabel = document.getElementById('top-p-label');
    const minPLabel = document.getElementById('min-p-label');

    // --- 2. STATE MANAGEMENT ---
    let allLogitContainers = [];

    // --- 3. SETUP EVENT LISTENERS ---

    // Tab Switching
    tabBtnMultiStream.addEventListener('click', () => setActiveTab('multi-stream'));
    tabBtnNgram.addEventListener('click', () => setActiveTab('ngram'));

    // Multi-Stream interactions (unchanged)
    mainPrompt.addEventListener("input", () => {
        const derivedPrompts = promptsContainer.querySelectorAll('.derived-prompt');
        derivedPrompts.forEach(p => p.remove());
    });
    halfCtxBtn.addEventListener("click", handleHalfContext);
    generateBtn.addEventListener("click", () => {
        streamsContainer.innerHTML = "";
        const prompts = document.querySelectorAll(".prompt-input");
        prompts.forEach((promptTextarea, index) => {
            if (promptTextarea.value.trim()) {
                setupStream(promptTextarea.value, index);
            }
        });
    });

    // Context Slice: "Probe" button
    ngramProbeBtn.addEventListener('click', handleProbeClick);

    // ⭐ NEW: Filter slider listeners
    [topKSlider, topPSlider, minPSlider].forEach(slider => {
        slider.addEventListener('input', applyAllFilters);
    });

    // --- 4. CORE FUNCTIONS ---

    async function handleProbeClick() {
        if (!ngramInput.value.trim()) return;
        ngramProbeBtn.disabled = true;
        ngramProbeBtn.textContent = "Probing...";
        filterControls.style.display = 'none';
        ngramResultsContainer.innerHTML = `<p class="placeholder">Tokenizing, running 4 inferences, and analyzing...</p>`;
        allLogitContainers = []; // Reset for new query

        try {
            const response = await fetch('/v1/probe_context_slices', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: ngramInput.value, n_probs: 1000 })
            });
            if (!response.ok) throw new Error(`Server error: ${response.status} ${await response.text()}`);
            
            const results = await response.json();
            if (!Array.isArray(results) || results.length === 0) throw new Error("Invalid response from server.");

            renderContextSliceResults(results);
            filterControls.style.display = 'grid'; // Show filters now that we have data
            applyAllFilters(); // Apply initial filter state

        } catch (error) {
            console.error("Context slice probe failed:", error);
            ngramResultsContainer.innerHTML = `<p class="error">${error.message}</p>`;
        } finally {
            ngramProbeBtn.disabled = false;
            ngramProbeBtn.textContent = "Probe Context Slices";
        }
    }
    
    // Renders the entire context slice results view
    function renderContextSliceResults(results) {
        ngramResultsContainer.innerHTML = ""; // Clear placeholder

        const longestContext = results.find(r => r.slice_factor === 1.0);
        const shortestContext = results[results.length - 1];
        
        if (longestContext && shortestContext) {
            const longContextTopTokens = new Set();
            let cumulativeProb = 0;
            for (const item of longestContext.logprobs) {
                if (cumulativeProb >= 0.9) break;
                longContextTopTokens.add(item.token);
                cumulativeProb += item.probability;
            }

            const shortSightedContinuations = shortestContext.logprobs.filter(
                item => !longContextTopTokens.has(item.token)
            );
            
            createLogitCard(
                'Short-Sighted Continuations',
                `Tokens predicted with the last <strong>${shortestContext.slice_factor * 100}%</strong> context that were NOT in the top 90% probability mass of the full context.`,
                shortSightedContinuations,
                { isShortSighted: true }
            );
        }
        
        const gridContainer = document.createElement('div');
        gridContainer.className = 'context-slice-grid';
        ngramResultsContainer.appendChild(gridContainer);
        
        results.forEach(result => {
            createLogitCard(
                `Context: Last ${result.slice_factor * 100}%`,
                result.prompt_slice,
                result.logprobs,
                { container: gridContainer, isPrompt: true }
            );
        });
    }

    function createLogitCard(title, description, logprobs, options = {}) {
        const card = document.createElement('div');
        card.className = options.isShortSighted ? 'context-slice-card short-sighted-card' : 'context-slice-card';
        
        let descriptionHtml = options.isPrompt 
            ? `<pre class="prompt-preview" title="${description}">${description.replace(/\n/g, '↵')}</pre>`
            : `<p class="explanation">${description}</p>`;

        card.innerHTML = `<h3>${title}</h3>${descriptionHtml}<div class="logits-display"></div>`;
        
        const container = options.container || ngramResultsContainer;
        container.appendChild(card);
        
        const displayTarget = card.querySelector('.logits-display');
        // ⭐ CRITICAL: Store the full, unfiltered data on the DOM element
        displayTarget.fullLogitData = logprobs;
        allLogitContainers.push(displayTarget);

        updateLogitDisplay(logprobs, displayTarget, { highlightWordBoundaries: true });
    }

    // ⭐ NEW: The core filtering logic
    function applyAllFilters() {
        const topK = parseInt(topKSlider.value, 10);
        const topP = parseFloat(topPSlider.value);
        const minP = parseFloat(minPSlider.value);

        // Update labels
        topKLabel.textContent = `Top-K: ${topK}`;
        topPLabel.textContent = `Top-P: ${topP.toFixed(2)}`;
        minPLabel.textContent = `Min-P: ${(minP * 100).toFixed(2)}%`;

        allLogitContainers.forEach(container => {
            const fullData = container.fullLogitData || [];
            if (fullData.length === 0) return;

            // 1. Apply Min-P filter first
            let filteredData = fullData.filter(item => item.probability >= minP);

            // 2. Apply Top-P filter
            let cumulativeProb = 0;
            const topPData = [];
            for (const item of filteredData) {
                if (cumulativeProb >= topP) break;
                topPData.push(item);
                cumulativeProb += item.probability;
            }
            filteredData = topPData;
            
            // 3. Apply Top-K filter last
            filteredData = filteredData.slice(0, topK);

            // Create a set of visible tokens for fast lookup
            const visibleTokens = new Set(filteredData.map(item => item.token));
            
            // Toggle visibility without re-rendering
            for (const child of container.children) {
                const token = child.dataset.token;
                child.classList.toggle('hidden-by-filter', !visibleTokens.has(token));
            }
        });
    }

    // Renders the initial list of logit items
    function updateLogitDisplay(logprobs, targetElement, options = {}) {
        targetElement.innerHTML = "";
        if (!logprobs || logprobs.length === 0) {
            targetElement.innerHTML = `<p class="placeholder">No logit data.</p>`;
            return;
        }
        const topProbability = logprobs[0].probability;

        logprobs.forEach(item => {
            const token = String(item.token); // Ensure token is a string
            const probability = item.probability;
            
            const logitItem = document.createElement("div");
            logitItem.className = "logit-item";
            // ⭐ CRITICAL: Add a data attribute for identification
            logitItem.dataset.token = token; 

            if (options.highlightWordBoundaries && (token.startsWith(' ') || token.startsWith('\n'))) {
                logitItem.classList.add('word-boundary-token');
            }
            
            logitItem.innerHTML = `
                <div class="token-label"><pre>'${token}'</pre></div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${(probability / topProbability) * 100}%">
                        ${(probability * 100).toFixed(2)}%
                    </div>
                </div>
            `;
            targetElement.appendChild(logitItem);
        });
    }

    // --- 5. UTILITY & UNCHANGED FUNCTIONS ---
    function setActiveTab(tabName) {
        paneMultiStream.classList.toggle('active', tabName !== 'ngram');
        tabBtnMultiStream.classList.toggle('active', tabName !== 'ngram');
        paneNgram.classList.toggle('active', tabName === 'ngram');
        tabBtnNgram.classList.toggle('active', tabName === 'ngram');
    }

    function setupStream(prompt, index) {
        const streamId = `stream-${index}`;
        const card = document.createElement("div");
        card.className = "stream-card";
        card.innerHTML = `<h2>Stream #${index + 1}</h2><div class="stream-content"><div id="text-${streamId}" class="generated-text"></div><div id="logits-${streamId}" class="logits-display"><p class="placeholder">Connecting...</p></div></div>`;
        streamsContainer.appendChild(card);
        const generatedTextDiv = document.getElementById(`text-${streamId}`);
        const logitsDisplayDiv = document.getElementById(`logits-${streamId}`);
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/generate`);
        ws.onopen = () => ws.send(JSON.stringify({ prompt, max_tokens: 200 }));
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.status === "done") return;
            generatedTextDiv.textContent += data.content;
            // Note: The simple stream viewer does not get the new filtering behavior.
            if (data.logprobs) {
                const tempContainer = document.createElement('div');
                updateLogitDisplay(data.logprobs, tempContainer);
                logitsDisplayDiv.innerHTML = tempContainer.innerHTML;
            }
        };
    }
    
    async function handleHalfContext() {
        const allPrompts = promptsContainer.querySelectorAll('.prompt-input');
        const sourcePrompt = allPrompts[allPrompts.length - 1];
        if (!sourcePrompt.value.trim()) return alert("Last prompt box is empty.");

        halfCtxBtn.disabled = true;
        halfCtxBtn.textContent = "Processing...";
        try {
            const tokenizeResponse = await fetch('/v1/tokenize', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: sourcePrompt.value })
            });
            if (!tokenizeResponse.ok) throw new Error("Tokenization failed");
            const tokenData = await tokenizeResponse.json();
            const midpoint = Math.floor(tokenData.tokens.length / 2);
            const secondHalfTokens = tokenData.tokens.slice(midpoint);

            const detokenizeResponse = await fetch('/v1/detokenize', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tokens: secondHalfTokens })
            });
            if (!detokenizeResponse.ok) throw new Error("Detokenization failed");
            const detokenizeData = await detokenizeResponse.json();

            const newTextarea = document.createElement("textarea");
            newTextarea.className = "prompt-input derived-prompt";
            newTextarea.rows = 4;
            newTextarea.value = detokenizeData.content;
            newTextarea.readOnly = true;
            promptsContainer.appendChild(newTextarea);
        } catch (error) {
            console.error("Half-context error:", error);
            alert(`An error occurred: ${error.message}`);
        } finally {
            halfCtxBtn.disabled = false;
            halfCtxBtn.textContent = "+ Half Context";
        }
    }
});