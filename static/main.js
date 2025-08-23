document.addEventListener("DOMContentLoaded", () => {
    // --- 1. DEFINE ALL DOM ELEMENTS ---
    const tabBtnMultiStream = document.getElementById('tab-btn-multi-stream');
    const tabBtnNgram = document.getElementById('tab-btn-ngram');
    const paneMultiStream = document.getElementById('pane-multi-stream');
    const paneNgram = document.getElementById('pane-ngram');
    const ngramInput = document.getElementById('ngram-input');
    const ngramProbeBtn = document.getElementById('ngram-probe-btn');
    const ngramResultsContainer = document.getElementById('ngram-results-container');
    
    // Sampling controls
    const processLogitToggle = document.getElementById('process-logit-toggle');
    const mmmSliderContainer = document.getElementById('mmm-slider-container');
    const mmmSlider = document.getElementById('mmm-slider');
    const mmmLabel = document.getElementById('mmm-label');
    const sampleLogitToggleGroup = document.getElementById('sample-logit-toggle-group');
    const sampleLogitToggle = document.getElementById('sample-logit-toggle');
    const autoregressToggleGroup = document.getElementById('autoregress-toggle-group');
    const autoregressToggle = document.getElementById('autoregress-toggle');
    const autoregressLabel = document.getElementById('autoregress-label');
    
    // Config Panels
    const autoregressPanel = document.getElementById('autoregress-panel');
    const maxContinuationInput = document.getElementById('max-continuation-input');
    const maxContinuationLabel = document.getElementById('max-continuation-label');

    // Filter controls & Sample Button
    const filterControls = document.getElementById('filter-controls');
    const sampleLogitContainer = document.getElementById('sample-logit-container');
    const sampleLogitBtn = document.getElementById('sample-logit-btn');
    const stopAutoregressBtn = document.getElementById('stop-autoregress-btn');
    
    // Sliders
    const numSlicesSlider = document.getElementById('num-slices-slider');
    const momentumPMassSlider = document.getElementById('momentum-p-mass-slider');
    const numSlicesLabel = document.getElementById('num-slices-label');
    const momentumPMassLabel = document.getElementById('momentum-p-mass-label');
    const topKSlider = document.getElementById('top-k-slider');
    const topPSlider = document.getElementById('top-p-slider');
    const minPSlider = document.getElementById('min-p-slider');
    const topKLabel = document.getElementById('top-k-label');
    const topPLabel = document.getElementById('top-p-label');
    const minPLabel = document.getElementById('min-p-label');

    // --- 2. STATE MANAGEMENT ---
    let allLogitContainers = [];
    let rawSliceData = [];
    let initialPrompt = "";
    let isAutoregressRunning = false;
    // Multi-Stream elements (for completeness)
    const promptsContainer = document.getElementById("prompts-container");
    const streamsContainer = document.getElementById("streams-container");
    const halfCtxBtn = document.getElementById("half-ctx-btn");
    const generateBtn = document.getElementById("generate-btn");


    // --- 3. EVENT LISTENERS ---
    tabBtnMultiStream.addEventListener('click', () => setActiveTab('multi-stream'));
    tabBtnNgram.addEventListener('click', () => setActiveTab('ngram'));
    
    ngramProbeBtn.addEventListener('click', () => {
        initialPrompt = ngramInput.value; // Store the prompt on manual probe
        runProbe();
    });

    // Toggle Logic
    processLogitToggle.addEventListener('change', handleToggleChange);
    sampleLogitToggle.addEventListener('change', handleToggleChange);
    autoregressToggle.addEventListener('change', handleToggleChange);

    sampleLogitBtn.addEventListener('click', handleSampleClick);
    stopAutoregressBtn.addEventListener('click', () => { 
        isAutoregressRunning = false; 
        setAutoregressUI(false);
    });

    maxContinuationInput.addEventListener('input', () => {
        maxContinuationLabel.textContent = `Max Continuation Length: ${maxContinuationInput.value}`;
    });

    // Slider Listeners
    mmmSlider.addEventListener('input', () => {
        mmmLabel.textContent = `Markov Momentum Multiplier: ${parseFloat(mmmSlider.value).toFixed(2)}`;
        if (processLogitToggle.checked) updateDisplay();
    });
    numSlicesSlider.addEventListener('input', () => {
        numSlicesLabel.textContent = `Number of Slices: ${numSlicesSlider.value}`;
    });
    momentumPMassSlider.addEventListener('input', () => {
        momentumPMassLabel.textContent = `P-Mass Filter: ${getPiecewisePMass().toFixed(4)}`;
        if (rawSliceData.length > 0) updateDisplay();
    });
    [topKSlider, topPSlider, minPSlider].forEach(slider => slider.addEventListener('input', applyAllFilters));
    
    // Multi-stream listeners
    if(halfCtxBtn) halfCtxBtn.addEventListener("click", handleHalfContext);
    if(generateBtn) generateBtn.addEventListener("click", handleGenerateStreams);

    // Initial UI setup
    handleToggleChange();
    numSlicesLabel.textContent = `Number of Slices: ${numSlicesSlider.value}`;
    momentumPMassLabel.textContent = `P-Mass Filter: ${getPiecewisePMass().toFixed(4)}`;
    mmmLabel.textContent = `Markov Momentum Multiplier: ${parseFloat(mmmSlider.value).toFixed(2)}`;
    maxContinuationLabel.textContent = `Max Continuation Length: ${maxContinuationInput.value}`;

    // --- 4. CORE LOGIC ---

    function handleToggleChange() {
        const isProcess = processLogitToggle.checked;
        const isSample = sampleLogitToggle.checked;
        const isAutoregress = autoregressToggle.checked;

        mmmSliderContainer.classList.toggle('disabled', !isProcess);
        sampleLogitToggleGroup.classList.toggle('disabled', !isProcess);
        if (!isProcess && sampleLogitToggle.checked) {
            sampleLogitToggle.checked = false;
        }
        
        const isNowSample = processLogitToggle.checked && sampleLogitToggle.checked;
        autoregressToggleGroup.classList.toggle('disabled', !isNowSample);
        sampleLogitContainer.classList.toggle('disabled', !isNowSample);
        if (!isNowSample && autoregressToggle.checked) {
            autoregressToggle.checked = false;
        }

        const isNowAutoregress = isNowSample && autoregressToggle.checked;
        autoregressPanel.style.display = isNowAutoregress ? 'block' : 'none';

        const canAutoregress = isProcess && isSample;
        autoregressLabel.textContent = canAutoregress ? '[ ! AUTOREGRESS ! ]' : '[ AUTOREGRESS ]';
        autoregressLabel.classList.toggle('hackerman', canAutoregress);

        if (rawSliceData.length > 0) {
            updateDisplay();
        }
    }
    
    async function handleSampleClick() {
        const processedCard = ngramResultsContainer.querySelector('.processed-logit-card');
        if (!processedCard) return;

        const chosenToken = performSampling(processedCard);
        if (!chosenToken) {
            if (isAutoregressRunning) {
                console.log("Sampling returned no token. Stopping autoregression.");
                isAutoregressRunning = false;
                setAutoregressUI(false);
            }
            return;
        }

        ngramInput.value += chosenToken;
        
        if (autoregressToggle.checked && isAutoregressRunning) {
            const continuationLength = ngramInput.value.length - initialPrompt.length;
            if (continuationLength >= parseInt(maxContinuationInput.value, 10)) {
                console.log("Max continuation length reached.");
                isAutoregressRunning = false;
                setAutoregressUI(false);
            } else {
                 await runProbe(); // Continue the loop
            }
        }
    }
    
    function performSampling(card) {
        const displayTarget = card.querySelector('.logits-display');
        const visibleItems = Array.from(displayTarget.querySelectorAll('.logit-item:not(.hidden-by-filter)'));
        if (visibleItems.length === 0) return null;

        const visibleLogits = displayTarget.fullLogitData.filter(logit => 
            visibleItems.some(item => item.dataset.token === logit.token)
        );

        const totalVisibleProb = visibleLogits.reduce((sum, item) => sum + item.probability, 0);
        if (totalVisibleProb === 0) return null;
        
        const normalizedLogits = visibleLogits.map(item => ({...item, prob: item.probability / totalVisibleProb }));

        const rand = Math.random();
        let cumulativeProb = 0;
        let chosenToken = normalizedLogits[normalizedLogits.length - 1].token;

        for (const item of normalizedLogits) {
            cumulativeProb += item.prob;
            if (rand <= cumulativeProb) {
                chosenToken = item.token;
                break;
            }
        }

        const chosenItemElement = displayTarget.querySelector(`.logit-item[data-token="${CSS.escape(chosenToken)}"]`);
        if (chosenItemElement) {
            chosenItemElement.classList.add('sampled');
            setTimeout(() => chosenItemElement.classList.remove('sampled'), 1500);
        }
        return chosenToken;
    }

    async function runProbe() {
        if (!ngramInput.value.trim()) return;
        
        const isStartingAutoregress = autoregressToggle.checked && !isAutoregressRunning;
        if (isStartingAutoregress) {
            isAutoregressRunning = true;
            setAutoregressUI(true);
            initialPrompt = ngramInput.value;
        }

        ngramProbeBtn.disabled = true;
        ngramProbeBtn.textContent = isAutoregressRunning ? "Running..." : "Probing...";
        filterControls.style.display = 'none';
        if(!isAutoregressRunning) ngramResultsContainer.innerHTML = `<p class="placeholder">Tokenizing and running inferences...</p>`;

        try {
            const response = await fetch('/v1/probe_context_slices', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: ngramInput.value, n_probs: 1000, slices: generateSlices() })
            });
            if (!response.ok) throw new Error(`Server error: ${response.status} ${await response.text()}`);
            
            rawSliceData = await response.json();
            if (!Array.isArray(rawSliceData) || rawSliceData.length === 0) throw new Error("Invalid response from server.");
            
            updateDisplay();
            filterControls.style.display = 'grid';

            if (isAutoregressRunning) {
                await new Promise(resolve => setTimeout(resolve, 50)); // Short delay for UI update
                handleSampleClick();
            }

        } catch (error) {
            ngramResultsContainer.innerHTML = `<p class="error">${error.message}</p>`;
            isAutoregressRunning = false; 
            setAutoregressUI(false);
        } finally {
            if (!isAutoregressRunning) {
                 ngramProbeBtn.disabled = false;
                 ngramProbeBtn.textContent = "Probe Context Slices";
            }
        }
    }

    function setAutoregressUI(isRunning) {
        ngramProbeBtn.style.display = isRunning ? 'none' : 'inline-block';
        stopAutoregressBtn.style.display = isRunning ? 'inline-block' : 'none';
        ngramInput.readOnly = isRunning;
    }

    function generateSlices() {
        const numSlices = parseInt(numSlicesSlider.value, 10);
        return Array.from({ length: numSlices }, (_, i) => Math.pow(0.5, i));
    }
    
    function updateDisplay() {
        ngramResultsContainer.innerHTML = "";
        allLogitContainers = [];

        const momentumData = calculateMomentum(rawSliceData);

        if (processLogitToggle.checked) {
            renderProcessedLogits(momentumData);
        } else {
            renderAnalysisDisplay(momentumData);
        }
        applyAllFilters();
    }
    
    // ⭐ FIX IS HERE ⭐
    function calculateMomentum(results) {
        if (!results || results.length < 2) return [];
    
        // Create maps for efficient lookup
        const longContextMap = new Map(results[0].logprobs.map(i => [i.token, i.probability]));
        const shortContextMap = new Map(results[results.length - 1].logprobs.map(i => [i.token, i.probability]));
        // Get a union of all tokens from both contexts
        const allTokens = new Set([...longContextMap.keys(), ...shortContextMap.keys()]);
    
        const topProbOverall = results[0].logprobs[0]?.probability || 1.0;
        let momentumData = [];
    
        // Calculate momentum for ALL tokens
        for (const token of allTokens) {
            const endProb = longContextMap.get(token) || 0;   // Probability from the longest context
            const startProb = shortContextMap.get(token) || 0; // Probability from the shortest context
            const momentum = endProb - startProb;
    
            momentumData.push({ token, probability: startProb, momentum, startProb, endProb, topProbOverall });
        }
        
        // Return the full, unfiltered data, sorted by absolute momentum for display ranking
        return momentumData.sort((a, b) => Math.abs(b.momentum) - Math.abs(a.momentum));
    }

    // ⭐ FIX IS HERE ⭐
    function renderAnalysisDisplay(momentumData) {
        // This function now performs filtering FOR DISPLAY PURPOSES ONLY.
        const momentumPMass = getPiecewisePMass();
        const longestContext = rawSliceData[0];
    
        // Determine the set of tokens in the top P-Mass of the longest context
        const longContextTopTokens = new Set();
        let cumulativeProb = 0;
        for (const item of longestContext.logprobs) {
            if (cumulativeProb >= momentumPMass) break;
            longContextTopTokens.add(item.token);
            cumulativeProb += item.probability;
        }
    
        // Filter the complete momentum data just for this card
        const filteredMomentumData = momentumData.filter(item => !longContextTopTokens.has(item.token));
    
        createLogitCard(
            'Contextual Momentum',
            `How a token's probability changes from the shortest to the full context. Tokens shown were NOT in the top ${(momentumPMass * 100).toFixed(1)}% of the full context. <strong class="pos">Green</strong> means its rank improved, <strong class="neg">Red</strong> means it worsened.`,
            filteredMomentumData, // Use the filtered data for this card
            { isMomentum: true, prepend: true }
        );

        // Render individual slice cards (this part is unchanged)
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

    function renderProcessedLogits(momentumData) {
        // This function receives the FULL, unfiltered momentum data
        const processedLogits = calculateProcessedLogits(momentumData);
        createLogitCard(
            'Processed Logit Distribution',
            `New distribution created by applying momentum to tokens within the ${(getPiecewisePMass() * 100).toFixed(1)}% P-Mass filter of the longest context.`,
            processedLogits,
            { isProcessed: true }
        );
    }
    
    function calculateProcessedLogits(momentumData) {
        // This function also receives the FULL, unfiltered momentum data
        if (rawSliceData.length < 2) return [];

        const mmm = parseFloat(mmmSlider.value);
        const fullContext = rawSliceData[0];
        // The momentumMap is now complete, containing data for all relevant tokens
        const momentumMap = new Map(momentumData.map(i => [i.token, i.momentum]));
        
        const pMassFilter = getPiecewisePMass();
        const tokensToModify = new Set();
        let cumulativeProb = 0;
        for (const item of fullContext.logprobs) {
            if (cumulativeProb >= pMassFilter) break;
            tokensToModify.add(item.token);
            cumulativeProb += item.probability;
        }

        let processedProbs = [];
        let totalProbSum = 0;

        for (const item of fullContext.logprobs) {
            let newProb = item.probability;
            // The check is now meaningful because momentumMap is complete
            if (tokensToModify.has(item.token)) {
                const momentum = momentumMap.get(item.token) || 0;
                newProb = item.probability + (mmm * momentum);
            }
            newProb = Math.max(0, newProb);
            processedProbs.push({ token: item.token, rawProb: newProb });
            totalProbSum += newProb;
        }
        
        if (totalProbSum === 0) return [];
        const finalLogits = processedProbs.map(item => ({
            token: item.token,
            probability: item.rawProb / totalProbSum
        }));

        return finalLogits.sort((a, b) => b.probability - a.probability);
    }
    
    // --- The rest of the utility functions are unchanged ---
    
    function getPiecewisePMass() {
        const x = parseFloat(momentumPMassSlider.value);
        const onethird = 1.0 / 3.0;
        const twothirds = 2.0 / 3.0;
        if (x < onethird) return (x / onethird) * 0.1;
        if (x < twothirds) return 0.1 + ((x - onethird) / onethird) * 0.8;
        return 0.9 + ((x - twothirds) / onethird) * 0.1;
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
        if (options.prepend) container.prepend(card);
        else container.appendChild(card);
        const displayTarget = card.querySelector('.logits-display');
        displayTarget.fullLogitData = logitData;
        allLogitContainers.push(displayTarget);
        updateLogitDisplay(logitData, displayTarget, { isMomentum: options.isMomentum });
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
            filteredData = topPData.slice(0, topK);
            const visibleTokens = new Set(filteredData.map(item => item.token));
            for (const child of container.children) {
                child.classList.toggle('hidden-by-filter', !visibleTokens.has(child.dataset.token));
            }
        });
    }

    function updateLogitDisplay(logprobs, targetElement, options = {}) {
        targetElement.innerHTML = "";
        if (!logprobs || logprobs.length === 0) {
            targetElement.innerHTML = `<p class="placeholder">No logit data.</p>`; return;
        }
        const topProbability = logprobs[0]?.probability || 1.0;
        logprobs.forEach(item => {
            const token = String(item.token);
            const probability = item.probability;
            const logitItem = document.createElement("div");
            logitItem.className = "logit-item";
            logitItem.dataset.token = token;
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
            if (data.status === "done") { ws.close(); return; }
            generatedTextDiv.textContent += data.content;
            if (data.logprobs) {
                const tempContainer = document.createElement('div');
                updateLogitDisplay(data.logprobs, tempContainer);
                logitsDisplayDiv.innerHTML = tempContainer.innerHTML;
            }
        };
        ws.onerror = (err) => { console.error("WebSocket Error:", err); };
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