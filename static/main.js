document.addEventListener("DOMContentLoaded", () => {
    // --- 1. DEFINE ALL DOM ELEMENTS ---
    const ngramInput = document.getElementById('ngram-input');
    const ngramProbeBtn = document.getElementById('ngram-probe-btn');
    const ngramResultsContainer = document.getElementById('ngram-results-container');
    // Sampling controls
    const processLogitToggle = document.getElementById('process-logit-toggle');
    const mmmSliderContainer = document.getElementById('mmm-slider-container');
    const mmmSlider = document.getElementById('mmm-slider');
    const mmmLabel = document.getElementById('mmm-label');
    // Probe parameter controls
    const numSlicesSlider = document.getElementById('num-slices-slider');
    const momentumPMassSlider = document.getElementById('momentum-p-mass-slider');
    const numSlicesLabel = document.getElementById('num-slices-label');
    const momentumPMassLabel = document.getElementById('momentum-p-mass-label');
    // Filter controls
    const filterControls = document.getElementById('filter-controls');
    const topKSlider = document.getElementById('top-k-slider');
    const topPSlider = document.getElementById('top-p-slider');
    const minPSlider = document.getElementById('min-p-slider');
    const topKLabel = document.getElementById('top-k-label');
    const topPLabel = document.getElementById('top-p-label');
    const minPLabel = document.getElementById('min-p-label');
    // Elements for the other tab
    const tabBtnMultiStream = document.getElementById('tab-btn-multi-stream');
    const tabBtnNgram = document.getElementById('tab-btn-ngram');
    const paneMultiStream = document.getElementById('pane-multi-stream');
    const paneNgram = document.getElementById('pane-ngram');
    const halfCtxBtn = document.getElementById("half-ctx-btn");
    const generateBtn = document.getElementById("generate-btn");
    const promptsContainer = document.getElementById("prompts-container");
    const streamsContainer = document.getElementById("streams-container");


    // --- 2. STATE MANAGEMENT ---
    let allLogitContainers = [];
    let rawSliceData = [];
    let isProcessLogitMode = false;

    // --- 3. SETUP EVENT LISTENERS ---
    ngramProbeBtn.addEventListener('click', handleProbeClick);
    processLogitToggle.addEventListener('change', handleModeToggle);
    mmmSlider.addEventListener('input', () => {
        mmmLabel.textContent = `[SAMPLE LOGIT] Markov Momentum Multiplier: ${parseFloat(mmmSlider.value).toFixed(2)}`;
        if (isProcessLogitMode) updateDisplay();
    });
    numSlicesSlider.addEventListener('input', () => {
        numSlicesLabel.textContent = `Number of Slices: ${numSlicesSlider.value}`;
    });
    momentumPMassSlider.addEventListener('input', () => {
        const pMass = getPiecewisePMass();
        momentumPMassLabel.textContent = `P-Mass Filter: ${pMass.toFixed(4)}`;
        if (rawSliceData.length > 0) updateDisplay();
    });
    [topKSlider, topPSlider, minPSlider].forEach(slider => {
        slider.addEventListener('input', applyAllFilters);
    });
    // Listeners for the other tab
    tabBtnMultiStream.addEventListener('click', () => setActiveTab('multi-stream'));
    tabBtnNgram.addEventListener('click', () => setActiveTab('ngram'));
    halfCtxBtn.addEventListener("click", handleHalfContext);
    generateBtn.addEventListener("click", handleGenerateStreams);
    
    // Initialize labels
    numSlicesLabel.textContent = `Number of Slices: ${numSlicesSlider.value}`;
    momentumPMassLabel.textContent = `P-Mass Filter: ${getPiecewisePMass().toFixed(4)}`;
    mmmLabel.textContent = `[SAMPLE LOGIT] Markov Momentum Multiplier: ${parseFloat(mmmSlider.value).toFixed(2)}`;

    // --- 4. CORE LOGIC ---

    function handleModeToggle() {
        isProcessLogitMode = processLogitToggle.checked;
        mmmSliderContainer.classList.toggle('disabled', !isProcessLogitMode);
        if (rawSliceData.length > 0) {
            updateDisplay();
        }
    }

    async function handleProbeClick() {
        if (!ngramInput.value.trim()) return;
        ngramProbeBtn.disabled = true;
        ngramProbeBtn.textContent = "Probing...";
        filterControls.style.display = 'none';
        ngramResultsContainer.innerHTML = `<p class="placeholder">Tokenizing, running inferences, and analyzing...</p>`;
        try {
            const numSlices = parseInt(numSlicesSlider.value, 10);
            const response = await fetch('/v1/probe_context_slices', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: ngramInput.value, n_probs: 1000, num_slices: numSlices })
            });
            if (!response.ok) throw new Error(`Server error: ${response.status} ${await response.text()}`);
            rawSliceData = await response.json();
            if (!Array.isArray(rawSliceData) || rawSliceData.length === 0) throw new Error("Invalid response from server.");
            updateDisplay(); // Main render call
            filterControls.style.display = 'grid';
        } catch (error) {
            ngramResultsContainer.innerHTML = `<p class="error">${error.message}</p>`;
        } finally {
            ngramProbeBtn.disabled = false;
            ngramProbeBtn.textContent = "Probe Context Slices";
        }
    }
    
    // Main display router
    function updateDisplay() {
        ngramResultsContainer.innerHTML = "";
        allLogitContainers = [];

        if (isProcessLogitMode) {
            renderProcessedLogits();
        } else {
            renderAnalysisDisplay();
        }
        applyAllFilters();
    }

    // Piecewise P-Mass slider transformation
    function getPiecewisePMass() {
        const x = parseFloat(momentumPMassSlider.value); // 0.0 to 1.0
        const onethird = 1.0 / 3.0;
        const twothirds = 2.0 / 3.0;
        
        if (x < onethird) {
            // Map [0, 0.33] -> [0, 0.1]
            return (x / onethird) * 0.1;
        } else if (x < twothirds) {
            // Map [0.33, 0.66] -> [0.1, 0.9]
            const segment_x = (x - onethird) / onethird; // 0..1
            return 0.1 + segment_x * 0.8;
        } else {
            // Map [0.66, 1.0] -> [0.9, 1.0]
            const segment_x = (x - twothirds) / onethird; // 0..1
            return 0.9 + segment_x * 0.1;
        }
    }
    
    // --- 5. DISPLAY & CALCULATION FUNCTIONS ---

    // Renders the original analysis view (Momentum + Slices)
    function renderAnalysisDisplay() {
        calculateAndRenderMomentum(rawSliceData);
        const gridContainer = document.createElement('div');
        gridContainer.className = 'context-slice-grid';
        ngramResultsContainer.appendChild(gridContainer);
        rawSliceData.forEach(result => {
            createLogitCard(
                `Context: Last ${(result.slice_factor * 100).toFixed(2)}%`,
                result.prompt_slice, result.logprobs,
                { container: gridContainer, isPrompt: true }
            );
        });
    }

    // Renders the new processed logit sampler view
    function renderProcessedLogits() {
        const processedLogits = calculateProcessedLogits();
        createLogitCard(
            'Processed Logit Distribution',
            `New distribution created by applying momentum to tokens within the ${Math.round(getPiecewisePMass() * 100)}% P-Mass filter of the shortest context.`,
            processedLogits,
            { isProcessed: true }
        );
    }
    
    // Core calculation for the sampler mode
    function calculateProcessedLogits() {
        if (rawSliceData.length < 2) return [];

        const mmm = parseFloat(mmmSlider.value);
        const pMassFilter = getPiecewisePMass();

        const fullContext = rawSliceData[0];
        const shortestContext = rawSliceData[rawSliceData.length - 1];

        const fullContextMap = new Map(fullContext.logprobs.map(i => [i.token, i.probability]));
        const shortestContextMap = new Map(shortestContext.logprobs.map(i => [i.token, i.probability]));

        const tokensToModify = new Set();
        let cumulativeProb = 0;
        for (const item of shortestContext.logprobs) {
            if (cumulativeProb >= pMassFilter) break;
            tokensToModify.add(item.token);
            cumulativeProb += item.probability;
        }

        let processedProbs = [];
        let totalProbSum = 0;
        for (const [token, fullProb] of fullContextMap.entries()) {
            let newProb = fullProb;
            if (tokensToModify.has(token)) {
                const shortProb = shortestContextMap.get(token) || 0;
                const momentum = fullProb - shortProb;
                newProb = fullProb + (mmm * momentum);
            }
            newProb = Math.max(0, newProb);
            processedProbs.push({ token, rawProb: newProb });
            totalProbSum += newProb;
        }
        
        if (totalProbSum === 0) return [];
        const finalLogits = processedProbs.map(item => ({
            token: item.token,
            probability: item.rawProb / totalProbSum
        }));

        return finalLogits.sort((a, b) => b.probability - a.probability);
    }

    // Calculates and renders the momentum view
    function calculateAndRenderMomentum(results) {
        // ⭐ FIX IS HERE: Changed getTransformedPMass() to getPiecewisePMass()
        const momentumPMass = getPiecewisePMass();

        const probMaps = results.map(r => 
            r.logprobs.reduce((map, item) => (map[item.token] = item.probability, map), {})
        );

        const longestContext = results[0];
        const shortestContext = results[results.length - 1];
        if (!shortestContext || !longestContext) return;

        const longContextTopTokens = new Set();
        let cumulativeProb = 0;
        for (const item of longestContext.logprobs) {
            if (cumulativeProb >= momentumPMass) break;
            longContextTopTokens.add(item.token);
            cumulativeProb += item.probability;
        }

        let momentumData = [];
        const topProbOverall = longestContext.logprobs[0].probability;

        for (const item of shortestContext.logprobs) {
            const token = item.token;
            if (longContextTopTokens.has(token)) continue;

            const startProb = item.probability;
            const endProb = probMaps[0][token] || 0;
            const momentum = endProb - startProb;
            
            momentumData.push({
                token, probability: startProb, momentum, startProb, endProb, topProbOverall
            });
        }
        
        momentumData.sort((a, b) => Math.abs(b.momentum) - Math.abs(a.momentum));

        createLogitCard(
            'Contextual Momentum',
            `How a token's probability changes from the shortest to the full context. Tokens shown were NOT in the top ${Math.round(momentumPMass * 100)}% of the full context. <strong class="pos">Green</strong> means its rank improved, <strong class="neg">Red</strong> means it worsened.`,
            momentumData,
            { isMomentum: true, prepend: true }
        );
    }


    function createLogitCard(title, description, logitData, options = {}) {
        const card = document.createElement('div');
        card.className = 'context-slice-card';
        if (options.isMomentum) card.classList.add('momentum-card');
        if (options.isProcessed) card.classList.add('processed-logit-card');

        let descriptionHtml = options.isPrompt 
            ? `<pre class="prompt-preview" title="${description}">${description.replace(/\n/g, '↵')}</pre>`
            : `<p class="explanation">${description}</p>`;
        card.innerHTML = `<h3>${title}</h3>${descriptionHtml}<div class="logits-display"></div>`;
        
        const container = options.container || ngramResultsContainer;
        if (options.prepend) {
            container.prepend(card);
        } else {
            container.appendChild(card);
        }
        
        const displayTarget = card.querySelector('.logits-display');
        displayTarget.fullLogitData = logitData;
        allLogitContainers.push(displayTarget);

        updateLogitDisplay(logitData, displayTarget, { 
            highlightWordBoundaries: true, isMomentum: options.isMomentum 
        });
    }

    function applyAllFilters() {
        const topK = parseInt(topKSlider.value, 10);
        const topP = parseFloat(topPSlider.value);
        const minP = parseFloat(minPSlider.value);

        topKLabel.textContent = `Top-K: ${topK}`;
        topPLabel.textContent = `Top-P: ${topP.toFixed(2)}`;
        minPLabel.textContent = `Min-P: ${(minP * 100).toFixed(2)}%`;

        allLogitContainers.forEach(container => {
            const fullData = container.fullLogitData || [];
            if (fullData.length === 0) return;
            
            let filteredData = fullData.filter(item => item.probability >= minP);

            let cumulativeProb = 0;
            const topPData = [];
            for (const item of filteredData) {
                if (cumulativeProb >= topP) break;
                topPData.push(item);
                cumulativeProb += item.probability;
            }
            filteredData = topPData;
            filteredData = filteredData.slice(0, topK);

            const visibleTokens = new Set(filteredData.map(item => item.token));
            for (const child of container.children) {
                child.classList.toggle('hidden-by-filter', !visibleTokens.has(child.dataset.token));
            }
        });
    }

    function updateLogitDisplay(logprobs, targetElement, options = {}) {
        targetElement.innerHTML = "";
        if (!logprobs || logprobs.length === 0) {
            targetElement.innerHTML = `<p class="placeholder">No logit data.</p>`;
            return;
        }
        const topProbability = logprobs[0]?.probability || 1.0;

        logprobs.forEach(item => {
            const token = String(item.token);
            const probability = item.probability;
            const logitItem = document.createElement("div");
            logitItem.className = "logit-item";
            logitItem.dataset.token = token;
            if (options.highlightWordBoundaries && (token.startsWith(' ') || token.startsWith('\n'))) {
                logitItem.classList.add('word-boundary-token');
            }
            const tokenLabel = document.createElement("div");
            tokenLabel.className = "token-label";
            const pre = document.createElement('pre');
            pre.textContent = `'${token}'`;
            tokenLabel.appendChild(pre);
            const probBarContainer = document.createElement("div");
            probBarContainer.className = "prob-bar-container";
            const probBar = document.createElement("div");
            probBar.className = "prob-bar";
            probBar.style.width = `${(probability / topProbability) * 100}%`;
            probBar.textContent = `${(probability * 100).toFixed(2)}%`;
            probBarContainer.appendChild(probBar);
            if (options.isMomentum && item.momentum !== undefined) {
                logitItem.dataset.momentumDir = item.momentum > 0 ? "positive" : "negative";
                probBarContainer.classList.add('has-momentum');
                const topScale = item.topProbOverall;
                const startPos = (item.startProb / topScale) * 100;
                const endPos = (item.endProb / topScale) * 100;
                probBarContainer.style.setProperty('--start-pos', `${Math.min(startPos, endPos)}%`);
                probBarContainer.style.setProperty('--end-pos', `${endPos}%`);
                probBarContainer.style.setProperty('--width', `${Math.abs(endPos - startPos)}%`);
            }
            logitItem.appendChild(tokenLabel);
            logitItem.appendChild(probBarContainer);
            targetElement.appendChild(logitItem);
        });
    }

    // --- 6. UTILITY & UNCHANGED FUNCTIONS ---
    function setActiveTab(tabName) {
        paneMultiStream.classList.toggle('active', tabName !== 'ngram');
        tabBtnMultiStream.classList.toggle('active', tabName !== 'ngram');
        paneNgram.classList.toggle('active', tabName === 'ngram');
        tabBtnNgram.classList.toggle('active', tabName === 'ngram');
    }
    
    function handleGenerateStreams() {
        streamsContainer.innerHTML = "";
        promptsContainer.querySelectorAll(".prompt-input").forEach((prompt, i) => {
            if (prompt.value.trim()) setupStream(prompt.value, i);
        });
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