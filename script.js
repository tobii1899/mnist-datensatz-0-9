const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const clearButton = document.getElementById("clearButton");
const predictButton = document.getElementById("predictButton");
const predictionNumber = document.getElementById("predictionNumber");
const results = document.getElementById("results");

let drawing = false;
let model = null;

async function loadModel() {
    try {
        predictButton.disabled = true;
        predictButton.innerHTML = "Loading Model...";
        
        model = await tf.loadLayersModel("./model_js/model.json");
        
        predictButton.disabled = false;
        predictButton.innerHTML = `Predict <span>→</span>`;
    } catch (error) {
        console.error(error);
        predictionNumber.className = "prediction-empty";
        predictionNumber.textContent = "Model Error";
    }
}

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = 18;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "white";

function getPosition(event) {
    const rect = canvas.getBoundingClientRect();

    return {
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height)
    };
}

canvas.addEventListener("pointerdown", event => {
    drawing = true;
    const position = getPosition(event);

    ctx.beginPath();
    ctx.moveTo(position.x, position.y);
});

canvas.addEventListener("pointermove", event => {
    if (!drawing) return;

    const position = getPosition(event);
    ctx.lineTo(position.x, position.y);
    ctx.stroke();
});

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

function stopDrawing() {
    drawing = false;
    ctx.closePath();
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    predictionNumber.className = "prediction-empty";
    predictionNumber.textContent = "Draw something";

    results.innerHTML = "";
}

async function predict() {
    if (!model) {
        alert("Model is not loaded yet!");
        return;
    }

    predictButton.disabled = true;
    predictButton.innerHTML = "Thinking...";

    try {
        const tensor = tf.tidy(() => {
            return tf.browser.fromPixels(canvas, 1)
                .resizeBilinear([28, 28])
                .toFloat()
                .div(255.0)
                .expandDims(0);
        });

        const predictionTensor = await model.predict(tensor);
        const probabilities = await predictionTensor.data();
        tensor.dispose();

        const predictions = Array.from(probabilities).map((prob, index) => ({
            digit: index,
            probability: prob
        })).sort((a, b) => b.probability - a.probability);

        const best = predictions[0];

        predictionNumber.className = "prediction-number";
        predictionNumber.textContent = best.digit;

        results.innerHTML = "";

        predictions.forEach(result => {
            const row = document.createElement("div");
            row.className = "result";

            row.innerHTML = `
                <div class="digit">
                    ${result.digit}
                </div>

                <div class="bar-container">
                    <div
                        class="bar"
                        style="width: ${result.probability * 100}%">
                    </div>
                </div>

                <div class="probability">
                    ${(result.probability * 100).toFixed(1)}%
                </div>
            `;

            results.appendChild(row);
        });

    } catch (error) {
        predictionNumber.className = "prediction-empty";
        predictionNumber.textContent = "Error";
        results.innerHTML = "";
        console.error(error);

    } finally {
        predictButton.disabled = false;
        predictButton.innerHTML = `
            Predict
            <span>→</span>
        `;
    }
}

clearButton.addEventListener("click", clearCanvas);
predictButton.addEventListener("click", predict);

loadModel();