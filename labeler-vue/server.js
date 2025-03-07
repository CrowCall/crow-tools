// server.js
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({
    limit: '200mb',
}));

// Serve static files from "public" and the project root
app.use(express.static('public'));
app.use(express.static('.'));

// Endpoint to update labels.json on disk
app.post('/updateLabels', (req, res) => {
  const update = req.body; // Expecting: { segmentKey, labels }
  const labelsFile = path.join(__dirname, 'public', 'labels.json');
  let allLabels = {};
  try {
    if (fs.existsSync(labelsFile)) {
      const data = fs.readFileSync(labelsFile, 'utf8');
      allLabels = JSON.parse(data);
    }
  } catch (e) {
    console.error("Error reading existing labels.json:", e);
  }
  allLabels[update.segmentKey] = update.labels;
  try {
    fs.writeFileSync(labelsFile, JSON.stringify(allLabels, null, 2));
    res.json({ success: true });
  } catch (e) {
    console.error("Error writing labels.json:", e);
    res.status(500).json({ success: false, error: 'Failed to update labels.' });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
});

// Generic error-handling middleware
app.use((err, req, res, next) => {
    console.error("Error occurred:", err);
    res.status(err.status || 500).json({
        error: err.message || "Internal Server Error"
    });
});
