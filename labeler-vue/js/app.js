const app = Vue.createApp({
    data() {
        return {
            segments: [],
            labels: {},
            csvData: {},
            currentPage: 1,
            segmentsPerPage: 50,
            playbackSpeed: 1,
            totalPagesCached: 0,
            errorMessage: ""
        };
    },
    computed: {
        totalPages() {
            return this.totalPagesCached;
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
            const isSegmentLabeled = (segKey) => {
                const lbl = this.labels[segKey];
                if (!lbl) return false;
                if (
                    lbl.crowCount !== null ||
                    lbl.crowAge !== null ||
                    lbl.quality !== null ||
                    lbl.alert === true ||
                    lbl.begging === true ||
                    lbl.grief === true ||
                    lbl.softSong === true ||
                    lbl.rattle === true ||
                    lbl.mob === true
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
        loadCrowsCSV() {
            const self = this;
            Papa.parse('/csv/crows.csv', {
                download: true,
                header: true,
                complete(results) {
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
                    this.loadCrowsCSV();
                    this.totalPagesCached = Math.ceil(this.segments.length / this.segmentsPerPage);
                })
                .catch(err => console.error('Error loading segments:', err));
        },
        attachCSVtoSegments() {
            this.segments.forEach(seg => {
                if (this.csvData[seg.id]) {
                    seg.recordist = this.csvData[seg.id].recorder;
                    seg.media_notes = this.csvData[seg.id].mediaNotes;
                } else {
                    seg.recordist = 'Unknown';
                    seg.media_notes = '';
                }
            });
        },
        loadLabels() {
            fetch('/auto_labels.json')
                .then(r => r.json())
                .then(data => {
                    this.labels = data;
                })
                .catch(err => {
                    console.warn('No auto_labels.json found, starting fresh.');
                    this.labels = {};
                });
        },
        saveLabelsToServer(payload) {
          // payload is expected to be: { segmentKey, labels }
          fetch('/updateLabels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          })
          .then(res => {
            if (!res.ok) {
              return res.json().then(data => { throw new Error(data.error || "Server error"); });
            }
            return res.json();
          })
          .then(data => {
            if (!data.success) {
              throw new Error("Failed to update labels on server.");
            }
          })
          .catch(err => {
            console.error('Error updating labels on server:', err);
            // Optionally, set a global error message.
          });
        },
        prevPage() {
            if (this.currentPage > 1) {
                this.currentPage--;
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        },
        nextPage() {
            if (this.currentPage < this.totalPages) {
                this.currentPage++;
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        },
        handleGotoPage(pageNum) {
            if (pageNum >= 1 && pageNum <= this.totalPages) {
                this.currentPage = pageNum;
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        },
        setPlaybackSpeed(speed) {
            this.playbackSpeed = speed;
        }
    },
    watch: {
        currentPage(newVal) {
            localStorage.setItem('currentPage', newVal);
        },
        playbackSpeed(newVal) {
            localStorage.setItem('playbackSpeed', newVal);
        }
    },
    mounted() {
        // Retrieve saved current page and playback speed from localStorage
        const savedPage = localStorage.getItem('currentPage');
        if (savedPage) {
            this.currentPage = parseInt(savedPage, 10);
        }
        const savedSpeed = localStorage.getItem('playbackSpeed');
        if (savedSpeed) {
            this.playbackSpeed = parseInt(savedSpeed, 10);
        }
        // Then load segments and labels as before
        this.loadSegments();
        this.loadLabels();
    }
});

app.component('pagination', window.Pagination);
app.component('segment-card', window.SegmentCard);

app.mount('#app');
