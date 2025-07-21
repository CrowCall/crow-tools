const app = Vue.createApp({
    data() {
        return {
            segments: [],
            // Only a single cached filtered set:
            filteredSegmentsCache: [],
            labels: {},
            csvData: {},
            currentPage: 1,
            segmentsPerPage: 30,
            playbackSpeed: 1,
            totalPagesCached: 0,
            errorMessage: "",
            datasetConfig: {},
            filters: {
                labelStatus: 'all',
                reviewStatus: 'all',
                callTypes: {
                    rattle: false,
                    softSong: false,
                    mob: false,
                    alert: false,
                    begging: false
                },
                crowCounts: {
                    0: false,
                    1: false,
                    2: false,
                    3: false,
                    4: false
                },
                crowAge: 'all',
                globalFilter: '',
                quality: 'all'
            },
            activeFilters: null,
            dataset: new URLSearchParams(window.location.search).get('dataset') || 'all-public'
        };
    },
    computed: {
        totalPages() {
            return Math.ceil(this.filteredSegmentsCache.length / this.segmentsPerPage);
        },
        paginatedSegments() {
            const start = (this.currentPage - 1) * this.segmentsPerPage;
            return this.filteredSegmentsCache.slice(start, start + this.segmentsPerPage);
        },
        // Bridge for legacy template references.
        filteredSegments() {
            return this.filteredSegmentsCache;
        },
        stats() {
            const uniqueFileIDs = new Set();
            // Use filtered segments if active filters are applied, otherwise use all segments.
            const segmentsToCount = this.activeFilters ? this.filteredSegmentsCache : this.segments;
            let totalSegments = segmentsToCount.length;
            let labeledSegments = 0;
            let totalDurationSec = 0;
            let labeledDurationSec = 0;
            const fileIDtoIsLabeled = {};
            const isSegmentLabeled = (segKey) => {
                const lbl = this.labels[segKey];
                if (!lbl) return false;
                return lbl.reviewed === true;
            };
            for (const seg of segmentsToCount) {
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
        // Rewritten filtering function using only the activeFilters object.
        applyFiltersToSegments() {
            return this.segments.filter(segment => {
                const segKey = this.segmentKey(segment);
                const labelData = this.labels[segKey];

                // Label Status Filter
                if (this.activeFilters.labelStatus !== 'all') {
                    const isLabeled = labelData && (labelData.crowCount !== undefined);
                    if (this.activeFilters.labelStatus === 'labeled' && !isLabeled) return false;
                    if (this.activeFilters.labelStatus === 'unlabeled' && isLabeled) return false;
                }

                // Review Status Filter
                if (this.activeFilters.reviewStatus !== 'all') {
                    const isReviewed = labelData && labelData.reviewed === true;
                    if (this.activeFilters.reviewStatus === 'reviewed' && !isReviewed) return false;
                    if (this.activeFilters.reviewStatus === 'unreviewed' && isReviewed) return false;
                }

                // Call Types Filter (including begging)
                if (labelData) {
                    const callTypeFilters = this.activeFilters.callTypes;
                    const hasActiveCallTypeFilters = Object.values(callTypeFilters).some(val => val);
                    if (hasActiveCallTypeFilters) {
                        const matchesAType =
                            (callTypeFilters.rattle && labelData.rattle) ||
                            (callTypeFilters.softSong && labelData.softSong) ||
                            (callTypeFilters.mob && labelData.mob) ||
                            (callTypeFilters.alert && labelData.alert) ||
                            (callTypeFilters.begging && labelData.begging);
                        if (!matchesAType) return false;
                    }
                }

                // Crow Count Filter
                if (labelData && labelData.crowCount !== undefined) {
                    const crowCountFilters = this.activeFilters.crowCounts;
                    const hasActiveCrowCountFilters = Object.values(crowCountFilters).some(val => val);
                    if (hasActiveCrowCountFilters && !crowCountFilters[labelData.crowCount]) {
                        return false;
                    }
                }

                // Crow Age Filter
                if (this.activeFilters.crowAge !== 'all' && labelData && labelData.crowAge !== undefined) {
                    if (this.activeFilters.crowAge === 'adult' && labelData.crowAge !== 1) return false;
                    if (this.activeFilters.crowAge === 'juvenile' && labelData.crowAge !== 2) return false;
                }

                // Quality Filter
                if (this.activeFilters.quality !== 'all') {
                    // require a defined quality that matches the filter
                    const q = labelData?.quality;
                    if (q == null || Number(q) !== Number(this.activeFilters.quality)) {
                        return false;
                    }
                }

                // Global Filter: searches across media_notes, recordist (author), file ID and cluster.
                if (this.activeFilters.globalFilter && this.activeFilters.globalFilter.trim() !== '') {
                    const searchTerm = this.activeFilters.globalFilter.trim().toLowerCase();
                    const mediaNotes = (segment.media_notes || '').toLowerCase();
                    const recordist = (segment.recordist || '').toLowerCase();
                    const fileId = segment.id.toString().toLowerCase();
                    const cluster = (segment.cluster !== undefined ? segment.cluster.toString() : '').toLowerCase();
                    if (
                        !mediaNotes.includes(searchTerm) &&
                        !recordist.includes(searchTerm) &&
                        !fileId.includes(searchTerm) &&
                        !cluster.includes(searchTerm)
                    ) {
                        return false;
                    }
                }

                return true;
            });
        },
        updateFilters(newFilters) {
            // Deep copy new filters.
            this.activeFilters = JSON.parse(JSON.stringify(newFilters));
            this.currentPage = 1;
            localStorage.setItem('activeFilters', JSON.stringify(this.activeFilters));

            // Recalculate the filtered segments.
            this.filteredSegmentsCache = this.applyFiltersToSegments();
        },
        // Initialize filteredSegmentsCache once segments are loaded.
        initializeFilterCache() {
            if (this.activeFilters) {
                this.filteredSegmentsCache = this.applyFiltersToSegments();
            } else {
                this.filteredSegmentsCache = this.segments;
            }
        },
        loadCrowsCSV() {
            const libs = this.datasetConfig.included_libraries || ['macaulay', 'xeno-canto'];
            const files = libs.map(l => `/cache/libraries/${l}/library.csv`);
            let remaining = files.length;
            files.forEach(file => {
                Papa.parse(file, {
                    download: true,
                    header: true,
                    complete: (results) => {
                        results.data.forEach(row => {
                            const id = row['ML Catalog Number'];
                            if (!id) return;
                            if (!this.csvData[id]) this.csvData[id] = {};
                            this.csvData[id].recorder = row['Recordist'] || this.csvData[id].recorder || 'Unknown';
                            this.csvData[id].mediaNotes = row['Media notes'] || this.csvData[id].mediaNotes || '';
                        });
                        if (!--remaining) this.attachCSVtoSegments();
                    },
                    error: err => {
                        console.error('Error parsing', file, err);
                        if (!--remaining) this.attachCSVtoSegments();
                    }
                });
            });
        },
        loadSegments() {
            fetch(`/cache/datasets/${this.dataset}/segments.json`)
                .then(r => r.json())
                .then(data => {
                    const segArray = [];
                    for (const [id, segs] of Object.entries(data)) {
                        segs.forEach(seg => {
                            seg.id = id;
                            if (!seg.library) seg.library = 'macaulay';
                            segArray.push(seg);
                        });
                    }
                    
                    // Sort by cluster
                    segArray.sort((a, b) => {
                        const clusterA = a.cluster || 1;
                        const clusterB = b.cluster || 1;
                        return clusterA - clusterB;
                    });
                    
                    this.segments = segArray;
                    this.loadCrowsCSV();
                    this.totalPagesCached = Math.ceil(this.segments.length / this.segmentsPerPage);
                    this.initializeFilterCache();
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
            fetch(`/cache/datasets/${this.dataset}/labels.json`)
                .then(r => r.json())
                .then(data => {
                    this.labels = data;
                    // Re-apply active filters now that labels have arrived
                    if (this.activeFilters) {
                        this.filteredSegmentsCache = this.applyFiltersToSegments();
                    }
                })
                .catch(err => {
                    console.warn('No labels json found, starting fresh.');
                    this.labels = {};
                });
        },
        saveLabelsToServer(payload) {
            fetch(`/updateLabels?dataset=${this.dataset}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
                .then(res => {
                    if (!res.ok) {
                        return res.json().then(data => {
                            throw new Error(data.error || "Server error");
                        });
                    }
                    return res.json();
                })
                .then(data => {
                    if (!data.success) {
                        throw new Error("Failed to update labels on server.");
                    }
                    const segmentToUpdate = this.segments.find(seg => this.segmentKey(seg) === payload.segmentKey);
                    if (segmentToUpdate) {
                        segmentToUpdate.labelData = payload.labels;
                    }
                    this.errorMessage = "";
                })
                .catch(err => {
                    console.error('Error updating labels on server:', err);
                    this.errorMessage = "Error updating labels: " + err.message;
                });
        },
        prevPage() {
            if (this.currentPage > 1) {
                this.currentPage--;
                window.scrollTo({top: 0, behavior: 'smooth'});
            }
        },
        nextPage() {
            if (this.currentPage < this.totalPages) {
                this.currentPage++;
                window.scrollTo({top: 0, behavior: 'smooth'});
            }
        },
        handleGotoPage(pageNum) {
            if (pageNum >= 1 && pageNum <= this.totalPages) {
                this.currentPage = pageNum;
                window.scrollTo({top: 0, behavior: 'smooth'});
            }
        },
        setPlaybackSpeed(speed) {
            this.playbackSpeed = speed;
        },
        // Delete segment handling remains the same.
        handleSegmentDeleted(segmentKey) {
            this.segments = this.segments.filter(seg => this.segmentKey(seg) !== segmentKey);
            this.filteredSegmentsCache = this.filteredSegmentsCache.filter(seg => this.segmentKey(seg) !== segmentKey);
            delete this.labels[segmentKey];
            this.deleteSegmentFromServer(segmentKey);
        },
        deleteSegmentFromServer(segmentKey) {
            fetch(`/deleteSegment?dataset=${this.dataset}`, {
                method: 'DELETE',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({segmentKey})
            })
                .then(res => res.json())
                .then(data => {
                    if (!data.success) {
                        throw new Error("Failed to delete segment on server.");
                    }
                })
                .catch(err => {
                    console.error('Error deleting segment on server:', err);
                    this.errorMessage = "Error deleting segment: " + err.message;
                });
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
        
        // Load saved filters if they exist
        const savedFilters = localStorage.getItem('activeFilters');
        if (savedFilters) {
            try {
                this.activeFilters = JSON.parse(savedFilters);
            } catch (e) {
                console.error('Error parsing saved filters:', e);
                this.activeFilters = null;
            }
        }
        
        // Now load dataset config then segments and labels
        fetch(`/cache/datasets/${this.dataset}/config.json`)
            .then(r => r.json())
            .then(cfg => { this.datasetConfig = cfg; })
            .finally(() => {
                this.loadSegments();
                this.loadLabels();
            });
    }
});

app.component('pagination', window.Pagination);
app.component('segment-card', window.SegmentCard);
app.component('filter-component', window.FilterComponent);
app.component('menu-component', window.MenuComponent);

// Custom directive for auto focus
app.directive("focus", {
  mounted(el) {
    el.focus();
  }
});

app.mount('#app');
