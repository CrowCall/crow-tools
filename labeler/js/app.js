const app = Vue.createApp({
    data() {
        return {
            segments: [],
            labels: {},
            csvData: {},
            currentPage: 1,
            segmentsPerPage: 30,
            playbackSpeed: 1,
            totalPagesCached: 0,
            errorMessage: "",
            visibleSegmentKeys: [], // Keep track of segments that should remain visible even if they don't match filters
            filters: {
                labelStatus: 'all',
                reviewStatus: 'all',
                callTypes: {
                    rattle: false,
                    softSong: false,
                    mob: false,
                    alert: false
                },
                crowCounts: {
                    0: false,
                    1: false,
                    2: false,
                    3: false,
                    4: false
                },
                crowAge: 'all',
                mediaNotesText: '',
                cluster: null
            },
            activeFilters: null
        };
    },
    computed: {
        totalPages() {
            return Math.ceil(this.filteredSegments.length / this.segmentsPerPage);
        },
        filteredSegments() {
            if (!this.activeFilters) return this.segments;
            
            return this.segments.filter(segment => {
                const segKey = `${segment.id}-${segment.start_time}-${segment.end_time}`;
                
                // If this segment is in our visible list, always include it
                if (this.visibleSegmentKeys.includes(segKey)) {
                    return true;
                }
                
                const labelData = this.labels[segKey];
                
                // Label status filter
                if (this.activeFilters.labelStatus !== 'all') {
                    const isLabeled = labelData && (labelData.crowCount !== undefined);
                    if (this.activeFilters.labelStatus === 'labeled' && !isLabeled) return false;
                    if (this.activeFilters.labelStatus === 'unlabeled' && isLabeled) return false;
                }
                
                // Review status filter
                if (this.activeFilters.reviewStatus !== 'all') {
                    const isReviewed = labelData && labelData.reviewed === true;
                    if (this.activeFilters.reviewStatus === 'reviewed' && !isReviewed) return false;
                    if (this.activeFilters.reviewStatus === 'unreviewed' && isReviewed) return false;
                }
                
                // Call types
                if (labelData) {
                    const callTypeFilters = this.activeFilters.callTypes;
                    // If any filters are active, the segment must match at least one
                    const hasActiveCallTypeFilters = Object.values(callTypeFilters).some(val => val);
                    
                    if (hasActiveCallTypeFilters) {
                        const matchesAType = 
                            (callTypeFilters.rattle && labelData.rattle) ||
                            (callTypeFilters.softSong && labelData.softSong) ||
                            (callTypeFilters.mob && labelData.mob) ||
                            (callTypeFilters.alert && labelData.alert);
                        
                        if (!matchesAType) return false;
                    }
                }
                
                // Crow count
                if (labelData && labelData.crowCount !== undefined) {
                    const crowCountFilters = this.activeFilters.crowCounts;
                    const hasActiveCrowCountFilters = Object.values(crowCountFilters).some(val => val);
                    
                    if (hasActiveCrowCountFilters && !crowCountFilters[labelData.crowCount]) {
                        return false;
                    }
                }
                
                // Crow age
                if (this.activeFilters.crowAge !== 'all' && labelData && labelData.crowAge !== undefined) {
                    if (this.activeFilters.crowAge === 'adult' && labelData.crowAge !== 1) return false;
                    if (this.activeFilters.crowAge === 'juvenile' && labelData.crowAge !== 2) return false;
                }
                
                // Media notes text
                if (this.activeFilters.mediaNotesText) {
                    // If there's a filter for media notes, require that media_notes exists and is not empty
                    if (!segment.media_notes || segment.media_notes.trim() === '') {
                        return false;
                    }
                    // Also check that it includes the filter text
                    if (!segment.media_notes.toLowerCase().includes(this.activeFilters.mediaNotesText.toLowerCase())) {
                        return false;
                    }
                }
                
                // Cluster
                if (this.activeFilters.cluster !== null && this.activeFilters.cluster !== '' && segment.cluster !== undefined) {
                    if (segment.cluster !== this.activeFilters.cluster) return false;
                }
                
                return true;
            });
        },
        paginatedSegments() {
            const start = (this.currentPage - 1) * this.segmentsPerPage;
            return this.filteredSegments.slice(start, start + this.segmentsPerPage);
        },
        // Stats aggregator
        stats() {
            const uniqueFileIDs = new Set();
            // Use filtered segments when filters are active, otherwise use all segments
            const segmentsToCount = this.activeFilters ? this.filteredSegments : this.segments;
            let totalSegments = segmentsToCount.length;
            let labeledSegments = 0;
            let totalDurationSec = 0;
            let labeledDurationSec = 0;
            const fileIDtoIsLabeled = {};
            const isSegmentLabeled = (segKey) => {
                const lbl = this.labels[segKey];
                if (!lbl) return false;
                if (
                    lbl.reviewed === true
                ) {
                    return true;
                }
                return false;
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
        loadCrowsCSV() {
          const files = ['/cache/csv/crows.csv', '/cache/csv/crows-xeno-canto.csv'];
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
            fetch('/cache/cluster_segments.json')
                .then(r => r.json())
                .then(data => {
                    const segArray = [];
                    for (const [id, segs] of Object.entries(data)) {
                        segs.forEach(seg => {
                            seg.id = id;
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
            fetch('/cache/cluster_labels.json')
                .then(r => r.json())
                .then(data => {
                    this.labels = data;
                })
                .catch(err => {
                    console.warn('No labels json found, starting fresh.');
                    this.labels = {};
                });
        },
        saveLabelsToServer(payload) {
          // payload is expected to be: { segmentKey, labels }
          
          // Make sure the segment stays visible even if it no longer matches the filter
          if (!this.visibleSegmentKeys.includes(payload.segmentKey)) {
            this.visibleSegmentKeys.push(payload.segmentKey);
          }
          
          fetch('/updateLabels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
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
            
            // Update the segment's labelData property to reflect the change
            const segmentToUpdate = this.segments.find(seg => 
              this.segmentKey(seg) === payload.segmentKey
            );
            
            if (segmentToUpdate) {
              segmentToUpdate.labelData = payload.labels;
            }
            
            // Clear any previous error message on success.
            this.errorMessage = "";
          })
          .catch(err => {
            console.error('Error updating labels on server:', err);
            this.errorMessage = "Error updating labels: " + err.message;
          });
        },
        prevPage() {
            if (this.currentPage > 1) {
                // Clear the visible segments list when changing pages
                this.visibleSegmentKeys = [];
                this.currentPage--;
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        },
        nextPage() {
            if (this.currentPage < this.totalPages) {
                // Clear the visible segments list when changing pages
                this.visibleSegmentKeys = [];
                this.currentPage++;
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        },
        handleGotoPage(pageNum) {
            if (pageNum >= 1 && pageNum <= this.totalPages) {
                // Clear the visible segments list when changing pages
                this.visibleSegmentKeys = [];
                this.currentPage = pageNum;
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        },
        setPlaybackSpeed(speed) {
            this.playbackSpeed = speed;
        },
        updateFilters(newFilters) {
            // Deep copy the filters to avoid reference issues
            this.activeFilters = JSON.parse(JSON.stringify(newFilters));
            
            // Reset to page 1 when filters change
            this.currentPage = 1;
            
            // Clear the list of visible segments on filter change
            this.visibleSegmentKeys = [];
            
            // Save filters to localStorage
            localStorage.setItem('activeFilters', JSON.stringify(this.activeFilters));
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
        
        // Now loadSegments and labels
        this.loadSegments();
        this.loadLabels();
    }
});

app.component('pagination', window.Pagination);
app.component('segment-card', window.SegmentCard);
app.component('filter-component', window.FilterComponent);

app.mount('#app');
