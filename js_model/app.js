// Get references to the video and canvas elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

// Load the COCO-SSD model
cocoSsd.load().then(model => {
    // Get access to the webcam stream
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            // Start detecting objects in the video frames
            detectFrame(video, model);
        };
    });
});

// Function to detect objects in the video frame
function detectFrame(video, model) {
    model.detect(video).then(predictions => {
        // Render the predictions on the canvas
        renderPredictions(predictions);
        requestAnimationFrame(() => {
            detectFrame(video, model);
        });
    });
}

// Function to render the predictions on the canvas
function renderPredictions(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        ctx.strokeStyle = "#00FF00"; // Green color
        ctx.lineWidth = 4; // 4 pixels wide
        // Draw the bounding box
        ctx.strokeRect(x, y, width, height);
        ctx.fillStyle = "#00FF00"; // Green color
        ctx.font = "18px Arial";
        // Draw the label with the class and score
        ctx.fillText(
            `${prediction.class} (${(prediction.score * 100).toFixed(2)}%)`,
            x,
            y > 10 ? y - 5 : 10
        );
    });
}
