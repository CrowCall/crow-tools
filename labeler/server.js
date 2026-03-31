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
app.use('/images', express.static(path.join(__dirname, '../docs/images')));
app.use(express.static('.'));

const CACHE_DIR = path.join(__dirname, '../.cache');
const DATASETS_DIR = path.join(CACHE_DIR, 'datasets');
const LIBRARIES_DIR = path.join(CACHE_DIR, 'libraries');
const DEFAULT_PUBLIC_LIBRARIES = ['macaulay', 'xeno-canto'];

function readJsonIfExists(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (e) {
    return fallback;
  }
}

function listLibraries() {
  try {
    return fs.readdirSync(LIBRARIES_DIR).filter(name => fs.statSync(path.join(LIBRARIES_DIR, name)).isDirectory());
  } catch (e) {
    return [];
  }
}

function loadDatasetConfig(datasetName) {
  const configPath = path.join(DATASETS_DIR, datasetName, 'config.json');
  const config = readJsonIfExists(configPath, { name: datasetName });
  if (!config.included_libraries || config.included_libraries.length === 0) {
    if (datasetName === 'all-public') {
      const publicLibraries = listLibraries().filter(name => !['backgrounds', 'local'].includes(name));
      config.included_libraries = publicLibraries.length ? publicLibraries : DEFAULT_PUBLIC_LIBRARIES;
    } else {
      config.included_libraries = [];
    }
  }
  config.selected_files = config.selected_files || {};
  return config;
}

function getSelectedFilesSet(config, libraryName) {
  const selected = config.selected_files && config.selected_files[libraryName];
  if (!selected) return null;
  return new Set(selected.map(String));
}

function isFileAllowed(config, libraryName, fileId) {
  const selectedSet = getSelectedFilesSet(config, libraryName);
  if (!selectedSet) return true;
  return selectedSet.has(String(fileId));
}

function getIncludedLibraries(datasetName) {
  return loadDatasetConfig(datasetName).included_libraries || [];
}

function findAudioInfo(datasetName, fileId) {
  const config = loadDatasetConfig(datasetName);
  for (const libraryName of config.included_libraries || []) {
    if (!isFileAllowed(config, libraryName, fileId)) continue;
    const audioDir = path.join(LIBRARIES_DIR, libraryName, 'audio');
    for (const ext of ['.mp3', '.wav']) {
      const candidate = path.join(audioDir, `${fileId}${ext}`);
      if (fs.existsSync(candidate)) {
        return {
          library: libraryName,
          path: candidate,
          ext,
          relativeUrl: `/cache/libraries/${libraryName}/audio/${fileId}${ext}`
        };
      }
    }
  }
  return null;
}

function getExcludedSegments(datasetName) {
  const excludedPath = path.join(DATASETS_DIR, datasetName, 'excluded_segments.json');
  return new Set(readJsonIfExists(excludedPath, []).map(String));
}

function getDatasetSegments(datasetName) {
  const config = loadDatasetConfig(datasetName);
  const datasetSegmentsPath = path.join(DATASETS_DIR, datasetName, 'segments.json');
  const excluded = getExcludedSegments(datasetName);
  const addIfAllowed = (target, fileId, segment, libraryName) => {
    const segmentKey = `${fileId}-${segment.start_time}-${segment.end_time}`;
    if (excluded.has(segmentKey)) return;
    if (!target[fileId]) target[fileId] = [];
    target[fileId].push({ ...segment, library: segment.library || libraryName });
  };

  if (fs.existsSync(datasetSegmentsPath)) {
    const datasetSegments = readJsonIfExists(datasetSegmentsPath, {});
    const filtered = {};
    for (const [fileId, segments] of Object.entries(datasetSegments)) {
      segments.forEach(segment => addIfAllowed(filtered, fileId, segment, segment.library || null));
    }
    return filtered;
  }

  const merged = {};
  for (const libraryName of config.included_libraries || []) {
    const segmentsPath = path.join(LIBRARIES_DIR, libraryName, 'segments.json');
    const librarySegments = readJsonIfExists(segmentsPath, {});
    for (const [fileId, segments] of Object.entries(librarySegments)) {
      if (!isFileAllowed(config, libraryName, fileId)) continue;
      segments.forEach(segment => addIfAllowed(merged, fileId, segment, libraryName));
    }
  }
  return merged;
}

// Return list of available datasets
app.get('/datasets', (req, res) => {
  try {
    const names = fs.readdirSync(DATASETS_DIR).filter(name => fs.statSync(path.join(DATASETS_DIR, name)).isDirectory());
    res.json({ datasets: names });
  } catch (e) {
    res.status(500).json({ error: 'Failed to list datasets' });
  }
});

// Return list of available libraries
app.get('/libraries', (req, res) => {
  try {
    const names = listLibraries();
    res.json({ libraries: names });
  } catch (e) {
    res.status(500).json({ error: 'Failed to list libraries' });
  }
});

app.get('/segments', (req, res) => {
  const dataset = req.query.dataset || 'all-public';
  try {
    res.json(getDatasetSegments(dataset));
  } catch (e) {
    console.error('Get segments error:', e);
    res.status(500).json({ error: 'Failed to load segments' });
  }
});

app.get('/audio/:dataset/:fileId', (req, res) => {
  const info = findAudioInfo(req.params.dataset, req.params.fileId);
  if (!info) {
    return res.status(404).json({ error: 'Audio not found for dataset' });
  }
  res.redirect(info.relativeUrl);
});

// Create a new dataset (optionally importing labels from another dataset)
app.post('/datasets', (req, res) => {
  const { name, included_libraries = [], importFrom } = req.body || {};
  if (!name) return res.status(400).json({ error: 'name required' });
  const newDir = path.join(DATASETS_DIR, name);
  try {
    if (fs.existsSync(newDir)) {
      return res.status(409).json({ error: 'Dataset already exists' });
    }
    if (!fs.existsSync(newDir)) fs.mkdirSync(newDir, { recursive: true });
    const cfg = { name, included_libraries };
    fs.writeFileSync(path.join(newDir, 'config.json'), JSON.stringify(cfg, null, 2));
    fs.writeFileSync(path.join(newDir, 'labels.json'), '{}');
    if (importFrom) {
      const src = path.join(DATASETS_DIR, importFrom, 'labels.json');
      if (fs.existsSync(src)) {
        const data = fs.readFileSync(src, 'utf8');
        fs.writeFileSync(path.join(newDir, 'labels.json'), data);
      }
    }
    res.json({ success: true });
  } catch (e) {
    console.error('Create dataset error:', e);
    res.status(500).json({ error: 'Failed to create dataset' });
  }
});

// Delete a dataset
app.delete('/datasets/:name', (req, res) => {
  const dir = path.join(DATASETS_DIR, req.params.name);
  try {
    fs.rmSync(dir, { recursive: true, force: true });
    res.json({ success: true });
  } catch (e) {
    console.error('Delete dataset error:', e);
    res.status(500).json({ error: 'Failed to delete dataset' });
  }
});

// Update dataset config or import labels
app.put('/datasets/:name', (req, res) => {
  const { included_libraries, importFrom } = req.body || {};
  const dir = path.join(DATASETS_DIR, req.params.name);
  const cfgFile = path.join(dir, 'config.json');
  try {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    let cfg = {};
    if (fs.existsSync(cfgFile)) cfg = JSON.parse(fs.readFileSync(cfgFile));
    if (Array.isArray(included_libraries)) cfg.included_libraries = included_libraries;
    if (!cfg.name) cfg.name = req.params.name;
    fs.writeFileSync(cfgFile, JSON.stringify(cfg, null, 2));
    const labelsFile = path.join(dir, 'labels.json');
    if (!fs.existsSync(labelsFile)) fs.writeFileSync(labelsFile, '{}');
    if (importFrom) {
      const src = path.join(DATASETS_DIR, importFrom, 'labels.json');
      const dest = labelsFile;
      if (fs.existsSync(src)) {
        let data = JSON.parse(fs.readFileSync(src));
        let destData = {};
        if (fs.existsSync(dest)) destData = JSON.parse(fs.readFileSync(dest));
        Object.assign(destData, data);
        fs.writeFileSync(dest, JSON.stringify(destData, null, 2));
      }
    }
    res.json({ success: true });
  } catch (e) {
    console.error('Update dataset error:', e);
    res.status(500).json({ error: 'Failed to update dataset' });
  }
});

// Endpoint to update labels.json on disk
app.post('/updateLabels', (req, res) => {
  const dataset = req.query.dataset || 'all-public';
  const update = req.body; // Expecting: { segmentKey, labels }
  const labelsFile = path.join(__dirname, '../', '.cache', 'datasets', dataset, 'labels.json');
  let allLabels = {};
  try {
    if (fs.existsSync(labelsFile)) {
      const data = fs.readFileSync(labelsFile, 'utf8');
      allLabels = JSON.parse(data);
    }
  } catch (e) {
    console.error("Error reading existing labels json:", e);
  }
  allLabels[update.segmentKey] = update.labels;
  try {
    fs.writeFileSync(labelsFile, JSON.stringify(allLabels, null, 4));
    res.json({ success: true });
  } catch (e) {
    console.error("Error writing labels json:", e);
    res.status(500).json({ success: false, error: 'Failed to update labels.' });
  }
});

// Add below your updateLabels endpoint in server.js
app.post('/updateNotationLabels', (req, res) => {
  const dataset = req.query.dataset || 'all-public';
  // Expecting: { fileId: 'xxx', notations: { <timestamp>: "transcription text", ... } }
  const update = req.body;
  const notationFile = path.join(__dirname, '../', '.cache', 'datasets', dataset, 'notation_labels.json');
  let allNotations = {};
  try {
    if (fs.existsSync(notationFile)) {
      const data = fs.readFileSync(notationFile, 'utf8');
      allNotations = JSON.parse(data);
    }
  } catch (e) {
    console.error("Error reading existing notation labels:", e);
  }

  // Update the notation for the file
  allNotations[update.fileId] = update.notations;

  try {
    fs.writeFileSync(notationFile, JSON.stringify(allNotations, null, 4));
    res.json({ success: true });
  } catch (e) {
    console.error("Error writing notation labels:", e);
    res.status(500).json({ success: false, error: 'Failed to update notations.' });
  }
});

app.get('/getNotationLabels', (req, res) => {
  const dataset = req.query.dataset || 'all-public';
  const fileId = req.query.fileId;
  const notationFile = path.join(__dirname, '../', '.cache', 'datasets', dataset, 'notation_labels.json');
  let allNotations = {};
  try {
    if (fs.existsSync(notationFile)) {
      const data = fs.readFileSync(notationFile, 'utf8');
      allNotations = JSON.parse(data);
    }
  } catch (e) {
    console.error("Error reading notation labels:", e);
    return res.status(500).json({ success: false, error: 'Error reading notation labels.' });
  }
  res.json({ success: true, notations: allNotations[fileId] || {} });
});

app.get('/api/embeddings', (req, res) => {
    const dataset = req.query.dataset;
    let embeddingsFile = path.join(__dirname, '../', '.cache', 'embeddings-3d.json');
    if (dataset) {
      const datasetEmbeddings = path.join(__dirname, '../', '.cache', 'datasets', dataset, 'embeddings-3d.json');
      if (fs.existsSync(datasetEmbeddings)) embeddingsFile = datasetEmbeddings;
    }
    if (fs.existsSync(embeddingsFile)) {
        fs.readFile(embeddingsFile, 'utf8', (err, data) => {
            if (err) {
                console.error("Error reading embeddings file:", err);
                return res.status(500).json({ success: false, error: err.message });
            }
            try {
                let embeddings = JSON.parse(data);
                if (dataset) {
                    const memo = new Map();
                    embeddings = embeddings.filter(item => {
                        const segmentKey = item.segment_key || '';
                        const fileId = segmentKey.split('-')[0];
                        if (!fileId) return false;
                        if (!memo.has(fileId)) {
                            memo.set(fileId, !!findAudioInfo(dataset, fileId));
                        }
                        return memo.get(fileId);
                    });
                }
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

// Endpoint to delete a segment and its label
app.delete('/deleteSegment', (req, res) => {
  const dataset = req.query.dataset || 'all-public';
  const { segmentKey } = req.body;

  // ----- Remove from labels file -----
  const labelsFile = path.join(__dirname, '../', '.cache', 'datasets', dataset, 'labels.json');
  let allLabels = {};
  try {
    if (fs.existsSync(labelsFile)) {
      const data = fs.readFileSync(labelsFile, 'utf8');
      allLabels = JSON.parse(data);
    }
  } catch(e) {
    console.error("Error reading labels json for deletion:", e);
  }
  if (allLabels[segmentKey]) {
    delete allLabels[segmentKey];
  }
  try {
    fs.writeFileSync(labelsFile, JSON.stringify(allLabels, null, 4));
  } catch(e) {
    console.error("Error writing labels json for deletion:", e);
    return res.status(500).json({ success: false, error: 'Failed to delete segment label.' });
  }

  // ----- Remove from segments file -----
  const segmentsFile = path.join(__dirname, '../', '.cache', 'datasets', dataset, 'segments.json');
  if (fs.existsSync(segmentsFile)) {
    let allSegments = {};
    try {
      const data = fs.readFileSync(segmentsFile, 'utf8');
      allSegments = JSON.parse(data);
    } catch(e) {
      console.error("Error reading segments json for deletion:", e);
    }

    const parts = segmentKey.split('-');
    const segId = parts[0];
    const segStart = Number(parts[1]);
    const segEnd = Number(parts[2]);

    if (allSegments[segId]) {
      allSegments[segId] = allSegments[segId].filter(seg => {
        return !(Number(seg.start_time) === segStart && Number(seg.end_time) === segEnd);
      });

      if (allSegments[segId].length === 0) {
        delete allSegments[segId];
      }
    }

    try {
      fs.writeFileSync(segmentsFile, JSON.stringify(allSegments, null, 4));
    } catch(e) {
      console.error("Error writing segments json for deletion:", e);
      return res.status(500).json({ success: false, error: 'Failed to delete segment from segments file.' });
    }
  } else {
    const excludedPath = path.join(__dirname, '../', '.cache', 'datasets', dataset, 'excluded_segments.json');
    let excluded = readJsonIfExists(excludedPath, []);
    if (!excluded.includes(segmentKey)) excluded.push(segmentKey);
    try {
      fs.writeFileSync(excludedPath, JSON.stringify(excluded, null, 2));
    } catch(e) {
      console.error("Error writing excluded segments json:", e);
      return res.status(500).json({ success: false, error: 'Failed to exclude segment.' });
    }
  }

  res.json({ success: true });
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
