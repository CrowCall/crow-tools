// js/components/filterComponent.js
window.FilterComponent = {
    props: ['filters', 'activeFilters'],
    template: `
    <div class="d-flex align-items-center">
        <button class="btn btn-sm" 
                :class="{'btn-outline-light': !hasActiveFilters, 'btn-primary': hasActiveFilters}"
                @click="openFilterModal" 
                data-bs-toggle="modal" 
                data-bs-target="#filterModal">
            <span style="font-size: 1.2rem;">🔍</span>
        </button>
        
        <!-- Filter Modal -->
        <div class="modal fade" id="filterModal" tabindex="-1" aria-labelledby="filterModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="filterModalLabel">Filter Segments</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Label Status -->
                        <div class="mb-3">
                            <label class="form-label">Label Status</label>
                            <div class="btn-group w-100" role="group">
                                <input type="radio" class="btn-check" name="labelStatus" id="all" value="all" v-model="localFilters.labelStatus">
                                <label class="btn btn-outline-primary" for="all">All</label>
                                <input type="radio" class="btn-check" name="labelStatus" id="labeled" value="labeled" v-model="localFilters.labelStatus">
                                <label class="btn btn-outline-primary" for="labeled">Labeled</label>
                                <input type="radio" class="btn-check" name="labelStatus" id="unlabeled" value="unlabeled" v-model="localFilters.labelStatus">
                                <label class="btn btn-outline-primary" for="unlabeled">Unlabeled</label>
                            </div>
                        </div>
                        
                        <!-- Review Status -->
                        <div class="mb-3">
                            <label class="form-label">Review Status</label>
                            <div class="btn-group w-100" role="group">
                                <input type="radio" class="btn-check" name="reviewStatus" id="allReview" value="all" v-model="localFilters.reviewStatus">
                                <label class="btn btn-outline-primary" for="allReview">All</label>
                                <input type="radio" class="btn-check" name="reviewStatus" id="reviewed" value="reviewed" v-model="localFilters.reviewStatus">
                                <label class="btn btn-outline-primary" for="reviewed">Reviewed</label>
                                <input type="radio" class="btn-check" name="reviewStatus" id="unreviewed" value="unreviewed" v-model="localFilters.reviewStatus">
                                <label class="btn btn-outline-primary" for="unreviewed">Unreviewed</label>
                            </div>
                        </div>
                        
                        <!-- Call Type (includes Begging) -->
                        <div class="mb-3">
                            <label class="form-label">Call Type</label>
                            <div class="d-flex flex-wrap" style="gap: 0.5rem;">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="rattleFilter" v-model="localFilters.callTypes.rattle">
                                    <label class="form-check-label" for="rattleFilter">Rattle</label>
                                </div>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="softsongFilter" v-model="localFilters.callTypes.softSong">
                                    <label class="form-check-label" for="softsongFilter">Softsong</label>
                                </div>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="mobFilter" v-model="localFilters.callTypes.mob">
                                    <label class="form-check-label" for="mobFilter">Mob</label>
                                </div>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="alertFilter" v-model="localFilters.callTypes.alert">
                                    <label class="form-check-label" for="alertFilter">Alert</label>
                                </div>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="beggingFilter" v-model="localFilters.callTypes.begging">
                                    <label class="form-check-label" for="beggingFilter">Begging</label>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Crow Count -->
                        <div class="mb-3">
                            <label class="form-label">Crow Count</label>
                            <div class="btn-group w-100" role="group">
                                <input type="checkbox" class="btn-check" id="count0" v-model="localFilters.crowCounts[0]">
                                <label class="btn btn-outline-primary" for="count0">0</label>
                                <input type="checkbox" class="btn-check" id="count1" v-model="localFilters.crowCounts[1]">
                                <label class="btn btn-outline-primary" for="count1">1</label>
                                <input type="checkbox" class="btn-check" id="count2" v-model="localFilters.crowCounts[2]">
                                <label class="btn btn-outline-primary" for="count2">2</label>
                                <input type="checkbox" class="btn-check" id="count3" v-model="localFilters.crowCounts[3]">
                                <label class="btn btn-outline-primary" for="count3">3</label>
                                <input type="checkbox" class="btn-check" id="count4" v-model="localFilters.crowCounts[4]">
                                <label class="btn btn-outline-primary" for="count4">Crowd</label>
                            </div>
                        </div>
                        
                        <!-- Crow Age -->
                        <div class="mb-3">
                            <label class="form-label">Crow Age</label>
                            <div class="btn-group w-100" role="group">
                                <input type="radio" class="btn-check" name="ageFilter" id="allAge" value="all" v-model="localFilters.crowAge">
                                <label class="btn btn-outline-primary" for="allAge">All</label>
                                <input type="radio" class="btn-check" name="ageFilter" id="adult" value="adult" v-model="localFilters.crowAge">
                                <label class="btn btn-outline-primary" for="adult">Adult</label>
                                <input type="radio" class="btn-check" name="ageFilter" id="juvenile" value="juvenile" v-model="localFilters.crowAge">
                                <label class="btn btn-outline-primary" for="juvenile">Juvenile</label>
                            </div>
                        </div>
                        
                        <!-- Quality -->
                        <div class="mb-3">
                            <label class="form-label">Quality</label>
                            <div class="btn-group w-100" role="group">
                                <input type="radio" class="btn-check" name="quality" id="quality_all" value="all" v-model="localFilters.quality">
                                <label class="btn btn-outline-primary" for="quality_all">All</label>
                                <input type="radio" class="btn-check" name="quality" id="quality1" value="1" v-model="localFilters.quality">
                                <label class="btn btn-outline-primary" for="quality1">Bad</label>
                                <input type="radio" class="btn-check" name="quality" id="quality2" value="2" v-model="localFilters.quality">
                                <label class="btn btn-outline-primary" for="quality2">Average</label>
                                <input type="radio" class="btn-check" name="quality" id="quality3" value="3" v-model="localFilters.quality">
                                <label class="btn btn-outline-primary" for="quality3">Best</label>
                            </div>
                        </div>
                        
                        <!-- Global Text Filter: Media Notes, Author, File ID or Cluster -->
                        <div class="mb-3">
                            <label for="globalFilter" class="form-label">Media Notes, Author, File ID or Cluster</label>
                            <div class="input-group">
                                <input type="text" class="form-control" id="globalFilter" placeholder="Search..." v-model="localFilters.globalFilter">
                                <button class="btn btn-outline-secondary" type="button" @click="localFilters.globalFilter = ''">Clear</button>
                            </div>
                        </div>
                        
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" @click="resetFilters">Reset Filters</button>
                        <button type="button" class="btn btn-primary" @click="applyFilters" data-bs-dismiss="modal">Apply Filters</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    `,
    data() {
        return {
            localFilters: this.initLocalFilters(),
        };
    },
    computed: {
        hasActiveFilters() {
            return this.activeFilters && (
                this.activeFilters.labelStatus !== 'all' ||
                this.activeFilters.reviewStatus !== 'all' ||
                Object.values(this.activeFilters.callTypes).some(val => val) ||
                Object.values(this.activeFilters.crowCounts).some(val => val) ||
                this.activeFilters.crowAge !== 'all' ||
                this.activeFilters.quality !== 'all' ||
                (this.activeFilters.globalFilter && this.activeFilters.globalFilter.trim() !== '')
            );
        }
    },
    methods: {
        initLocalFilters() {
            return {
                labelStatus: this.filters?.labelStatus || 'all',
                reviewStatus: this.filters?.reviewStatus || 'all',
                callTypes: {
                    rattle: this.filters?.callTypes?.rattle || false,
                    softSong: this.filters?.callTypes?.softSong || false,
                    mob: this.filters?.callTypes?.mob || false,
                    alert: this.filters?.callTypes?.alert || false,
                    begging: this.filters?.callTypes?.begging || false
                },
                crowCounts: {
                    0: this.filters?.crowCounts?.[0] || false,
                    1: this.filters?.crowCounts?.[1] || false,
                    2: this.filters?.crowCounts?.[2] || false,
                    3: this.filters?.crowCounts?.[3] || false,
                    4: this.filters?.crowCounts?.[4] || false
                },
                crowAge: this.filters?.crowAge || 'all',
                quality: this.filters?.quality || 'all',
                globalFilter: this.filters?.globalFilter || ''
            };
        },
        openFilterModal() {
            // Modal is opened via Bootstrap data attributes
        },
        applyFilters() {
            this.$emit('filters-changed', this.localFilters);
        },
        resetFilters() {
            this.localFilters = {
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
                quality: 'all',
                globalFilter: ''
            };
            this.$emit('filters-changed', this.localFilters);
        }
    },
    watch: {
        filters: {
            handler(newFilters) {
                this.localFilters = this.initLocalFilters();
            },
            deep: true
        }
    }
};