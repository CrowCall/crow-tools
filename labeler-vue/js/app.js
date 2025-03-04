// js/app.js
const app = Vue.createApp({
    data() {
        return {
            segments: [],
            labels: {},
            csvData: {},  // Will map ID => { recorder, mediaNotes }
            currentPage: 1,
            segmentsPerPage: 50
        };
    },
    computed: {
        totalPages() {
            return Math.ceil(this.segments.length / this.segmentsPerPage);
        },
        paginatedSegments() {
            const start = (this.currentPage - 1) * this.segmentsPerPage;
            return this.segments.slice(start, start + this.segmentsPerPage);
        },
        // Stats aggregator
        stats() {
            const uniqueFileIDs = new Set();
            let totalSegments = this.segments.length;
            let labeledSegments = 0;

            let totalDurationSec = 0;
            let labeledDurationSec = 0;

            const fileIDtoIsLabeled = {};

            // Helper: Is a segment labeled?
            const isSegmentLabeled = (segKey) => {
                const lbl = this.labels[segKey];
                if (!lbl) return false;
                // If crowCount/crowAge is set, or any feature is true, or notes is non-empty
                if (
                    lbl.crowCount !== '' ||
                    lbl.crowAge !== '' ||
                    lbl.begging ||
                    lbl.softSong ||
                    lbl.badQuality ||
                    (lbl.notes && lbl.notes.trim() !== '')
                ) {
                    return true;
                }
                return false;
            };

            for (const seg of this.segments) {
                uniqueFileIDs.add(seg.id);
                const segKey = `${seg.id}-${seg.start_time}-${seg.end_time}`;
                const durationSec = seg.end_time - seg.start_time;
                totalDurationSec += durationSec;

                if (isSegmentLabeled(segKey)) {
                    labeledSegments++;
                    labeledDurationSec += durationSec;
                    fileIDtoIsLabeled[seg.id] = true;
                }
            }

            const totalFiles = uniqueFileIDs.size;
            let labeledFiles = 0;
            for (const fileID of uniqueFileIDs) {
                if (fileIDtoIsLabeled[fileID]) labeledFiles++;
            }

            const totalHours = totalDurationSec / 3600;
            const labeledHours = labeledDurationSec / 3600;

            // Percentages
            const filesPct = totalFiles === 0 ? 0 : Math.round((labeledFiles / totalFiles) * 100);
            const segmentsPct = totalSegments === 0 ? 0 : Math.round((labeledSegments / totalSegments) * 100);
            const hoursPct = totalHours === 0 ? 0 : Math.round((labeledHours / totalHours) * 100);

            return {
                labeledFiles,
                totalFiles,
                filesPct,

                labeledSegments,
                totalSegments,
                segmentsPct,

                labeledHours,
                totalHours,
                hoursPct
            };
        }
    },
    methods: {
        segmentKey(segment) {
            return `${segment.id}-${segment.start_time}-${segment.end_time}`;
        },
        // 1) Load CSV with Papa Parse
        loadCrowsCSV() {
            const self = this;
            Papa.parse('/crows.csv', {
                download: true,
                header: true,
                complete(results) {
                    // Build a dictionary keyed by the ID in "ML Catalog Number"
                    // Adjust if your CSV column names differ
                    results.data.forEach(row => {
                        const id = row['ML Catalog Number'];
                        if (id) {
                            self.csvData[id] = {
                                recorder: row['Recordist'] || 'Unknown',
                                mediaNotes: row['Media notes'] || ''
                            };
                        }
                    });
                    self.attachCSVtoSegments();
                },
                error(err) {
                    console.error('Error parsing crows.csv:', err);
                }
            });
        },
        // 2) Flatten segments.json
        loadSegments() {
            fetch('/segments.json')
                .then(r => r.json())
                .then(data => {
                    const segArray = [];
                    for (const [id, segs] of Object.entries(data)) {
                        segs.forEach(seg => {
                            seg.id = id;
                            segArray.push(seg);
                        });
                    }
                    this.segments = segArray;
                    // After we have segments, load the CSV so we can attach data
                    this.loadCrowsCSV();
                })
                .catch(err => console.error('Error loading segments:', err));
        },
        // 3) Attach CSV data to each segment
        attachCSVtoSegments() {
            // For each segment, if we find a matching ID in csvData, attach the fields
            this.segments.forEach(seg => {
                if (this.csvData[seg.id]) {
                    seg.recordist = this.csvData[seg.id].recorder;
                    seg.media_notes = this.csvData[seg.id].mediaNotes;
                } else {
                    // If not found, fallback
                    seg.recordist = 'Unknown';
                    seg.media_notes = '';
                }
            });
        },
        // Load labels.json
        loadLabels() {
            fetch('/labels.json')
                .then(r => r.json())
                .then(data => {
                    this.labels = data;
                })
                .catch(err => {
                    console.warn('No labels.json found, starting fresh.');
                    this.labels = {};
                });
        },
        // Save updated labels to the server
        saveLabelsToServer() {
            fetch('/updateLabels', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.labels)
            })
                .then(res => res.json())
                .then(data => {
                    if (!data.success) {
                        console.error('Failed to update labels on server.');
                    }
                })
                .catch(err => console.error('Error updating labels on server:', err));
        },
        // Pagination
        prevPage() {
            if (this.currentPage > 1) this.currentPage--;
        },
        nextPage() {
            if (this.currentPage < this.totalPages) this.currentPage++;
        },
        // Called when the user selects a page from the dropdown
        handleGotoPage(pageNum) {
            if (pageNum >= 1 && pageNum <= this.totalPages) {
                this.currentPage = pageNum;
            }
        }
    },
    mounted() {
        // 1) Load segments
        this.loadSegments();
        // 2) Load labels
        this.loadLabels();
    }
});

// Register child components
app.component('pagination', window.Pagination);
app.component('segment-card', window.SegmentCard);

app.mount('#app');
