// server.js
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({
    limit: '200mb',
}));

// Serve static files from "public" and the project root
app.use('/cache', express.static(path.join(__dirname, '../.cache')));
app.use(express.static('.'));

// Endpoint to update labels.json on disk
app.post('/updateLabels', (req, res) => {
  const update = req.body; // Expecting: { segmentKey, labels }
  const labelsFile = path.join(__dirname, '../', '.cache', 'auto_labels.json');
  let allLabels = {};
  try {
    if (fs.existsSync(labelsFile)) {
      const data = fs.readFileSync(labelsFile, 'utf8');
      allLabels = JSON.parse(data);
    }
  } catch (e) {
    console.error("Error reading existing auto_labels.json:", e);
  }
  allLabels[update.segmentKey] = update.labels;
  try {
    fs.writeFileSync(labelsFile, JSON.stringify(allLabels, null, 4));
    res.json({ success: true });
  } catch (e) {
    console.error("Error writing auto_labels.json:", e);
    res.status(500).json({ success: false, error: 'Failed to update labels.' });
  }
});

app.get('/api/embeddings', (req, res) => {
    const embeddingsFile = path.join(__dirname, '../', '.cache', 'embeddings-3d.json');
    if (fs.existsSync(embeddingsFile)) {
        fs.readFile(embeddingsFile, 'utf8', (err, data) => {
            if (err) {
                console.error("Error reading embeddings file:", err);
                return res.status(500).json({ success: false, error: err.message });
            }
            try {
                const embeddings = JSON.parse(data);
                res.json({ success: true, embeddings });
            } catch (parseErr) {
                console.error("Error parsing embeddings JSON:", parseErr);
                res.status(500).json({ success: false, error: parseErr.message });
            }
        });
    } else {
        res.status(404).json({ success: false, error: "Embeddings file not found" });
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
