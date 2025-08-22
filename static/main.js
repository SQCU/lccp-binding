document.addEventListener("DOMContentLoaded", () => {
    const generateBtn = document.getElementById("generate-btn");
    const addPromptBtn = document.getElementById("add-prompt-btn");
    const promptsContainer = document.getElementById("prompts-container");
    const streamsContainer = document.getElementById("streams-container");

    // --- Logic to add a new prompt textarea ---
    addPromptBtn.addEventListener("click", () => {
        const promptCount = promptsContainer.children.length + 1;
        const newTextarea = document.createElement("textarea");
        newTextarea.className = "prompt-input";
        newTextarea.rows = 3;
        newTextarea.placeholder = `Enter prompt #${promptCount}...`;
        promptsContainer.appendChild(newTextarea);
    });

    // --- Main generation logic ---
    generateBtn.addEventListener("click", () => {
        // Clear previous results
        streamsContainer.innerHTML = "";
        
        const prompts = document.querySelectorAll(".prompt-input");

        prompts.forEach((promptTextarea, index) => {
            const prompt = promptTextarea.value;
            if (prompt.trim()) {
                // For each valid prompt, create a UI card and a WebSocket connection
                setupStream(prompt, index);
            }
        });
    });

    /**
     * Creates the UI and WebSocket connection for a single generation stream.
     * @param {string} prompt - The text prompt to send.
     * @param {number} index - The unique index for this stream.
     */
    function setupStream(prompt, index) {
        const streamId = `stream-${index}`;
        
        // 1. Create the output card for this stream
        const card = document.createElement("div");
        card.className = "stream-card";
        card.innerHTML = `
            <h2>Stream #${index + 1}</h2>
            <div class="stream-content">
                <div id="text-${streamId}" class="generated-text"></div>
                <div id="logits-${streamId}" class="logits-display">
                    <p class="placeholder">Connecting...</p>
                </div>
            </div>
        `;
        streamsContainer.appendChild(card);

        // 2. Get references to this card's specific elements
        const generatedTextDiv = document.getElementById(`text-${streamId}`);
        const logitsDisplayDiv = document.getElementById(`logits-${streamId}`);

        // 3. Establish a dedicated WebSocket for this stream
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/generate`);

        ws.onopen = () => {
            console.log(`WebSocket for stream ${index} connected.`);
            logitsDisplayDiv.querySelector('.placeholder').textContent = 'Sending prompt...';
            ws.send(JSON.stringify({ prompt: prompt, max_tokens: 200 }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                console.error(`Error from server on stream ${index}:`, data.error);
                logitsDisplayDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                return;
            }
            
            if (data.status === "done") {
                console.log(`Stream ${index} finished.`);
                return;
            }

            // Append text and update logits for THIS specific card
            generatedTextDiv.textContent += data.content;
            if (data.logprobs) {
                updateLogitDisplay(data.logprobs, logitsDisplayDiv);
            }
        };

        ws.onclose = () => {
            console.log(`WebSocket for stream ${index} disconnected.`);
            logitsDisplayDiv.innerHTML += '<p class="placeholder">Connection closed.</p>';
        };

        ws.onerror = (error) => {
            console.error(`WebSocket error on stream ${index}:`, error);
            logitsDisplayDiv.innerHTML = `<p class="error">Connection failed.</p>`;
        };
    }

    /**
     * Renders the logit visualization inside a specific element.
     * @param {Array} logprobs - The array of logit data.
     * @param {HTMLElement} targetElement - The div where the visualization should be rendered.
     */
    function updateLogitDisplay(logprobs, targetElement) {
        targetElement.innerHTML = ""; // Clear previous logits
        if (!logprobs || logprobs.length === 0) return;

        const topProbability = logprobs[0].probability;

        logprobs.forEach(item => {
            const token = item.token;
            const probability = item.probability;
            if (token === null || typeof token === 'undefined') return;

            const tokenStr = JSON.stringify(token).slice(1, -1);
            const percentage = (probability * 100).toFixed(2);
            const barWidth = (probability / topProbability) * 100;

            const logitItem = document.createElement("div");
            logitItem.className = "logit-item";
            logitItem.innerHTML = `
                <div class="token-label">'${tokenStr}'</div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${barWidth}%;">${percentage}%</div>
                </div>
            `;
            targetElement.appendChild(logitItem);
        });
    }
});