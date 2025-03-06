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
    const labels = req.body;
    const labelsFile = path.join(__dirname, 'public', 'labels.json');
    fs.writeFile(labelsFile, JSON.stringify(labels, null, 2), (err) => {
        if (err) {
            console.error('Error writing labels.json:', err);
            return res.status(500).json({ success: false, error: 'Failed to update labels.' });
        }
        res.json({ success: true });
    });
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