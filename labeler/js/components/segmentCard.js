window.SegmentCard = {
  props: ['segment', 'labels', 'playbackSpeed', 'prevLabels'],
  template: `
    <div :class="['card', 'mb-2', 'p-2', cardBorderClass]" :style="leftBorderStyle">
      <!-- 1) Title Row -->
      <div class="d-flex flex-wrap align-items-center mb-1">
        <strong class="me-2">
          Recorder: {{ segment.recordist }} / ID: {{ segment.id }}
        </strong>
        <small class="text-muted">
          Time: {{ segment.start_time }} - {{ segment.end_time }},
          Conf: {{ segment.confidence.toFixed(2) }}
        </small>
      </div>

      <!-- 2) Media Notes -->
      <div class="mb-2" style="overflow: auto; text-overflow: ellipsis; max-height: 3em; font-size: .85em;">
        <div class="text-muted" :title="segment.media_notes">{{ segment.media_notes }}</div>
      </div>

      <!-- 3) Play Button, Slider, and Copy Button -->
      <div class="d-flex align-items-center flex-grow-1 mb-2" style="gap: 0.5rem;">
        <button :class="['btn', 'btn-sm', playButtonClass]" style="width: 60px;" @click="handlePlay">
          <span v-if="!isPlaying">Play</span>
          <span v-else>Pause</span>
        </button>
        <input type="range" class="form-range"
               style="margin: 0;"
               :min="segment.start_time"
               :max="segment.end_time"
               :step="0.1"
               v-model.number="currentTime"
               @input="onSliderChange">
        <small class="text-nowrap">
          {{ formattedTime(currentTime - segment.start_time) }} /
          {{ formattedTime(segment.end_time - segment.start_time) }}
        </small>
        <button class="btn btn-sm btn-outline-secondary" style="width: 40px;" @click="copyPrevious" :disabled="!prevLabels">
          📋
        </button>
      </div>
      
      <!-- Top row: Crow Count, Crow Age, Behavior, and Quality -->
      <div class="d-flex flex-wrap align-items-center justify-content-evenly" style="gap: 0.5rem;">
        <!-- Crow Count Group -->
        <div class="btn-group btn-group-sm flex-fill mx-1" role="group">
          <input type="radio" class="btn-check"
                 :name="'crowCount' + segmentKey"
                 :id="'crowCount1_' + segmentKey"
                 :value="1"
                 v-model.number="currentLabels.crowCount"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowCount1_' + segmentKey" title="1 Crow">1</label>
      
          <input type="radio" class="btn-check"
                 :name="'crowCount' + segmentKey"
                 :id="'crowCount2_' + segmentKey"
                 :value="2"
                 v-model.number="currentLabels.crowCount"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowCount2_' + segmentKey" title="2 Crows">2</label>
      
          <input type="radio" class="btn-check"
                 :name="'crowCount' + segmentKey"
                 :id="'crowCount3_' + segmentKey"
                 :value="3"
                 v-model.number="currentLabels.crowCount"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowCount3_' + segmentKey" title="3 Crows">3</label>
      
          <input type="radio" class="btn-check"
                 :name="'crowCount' + segmentKey"
                 :id="'crowCount4_' + segmentKey"
                 :value="4"
                 v-model.number="currentLabels.crowCount"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowCount4_' + segmentKey" title="Crowd">📣</label>
        </div>
      
        <!-- Crow Age Group -->
        <div class="btn-group btn-group-sm flex-fill mx-1" role="group">
          <input type="radio" class="btn-check"
                 :name="'crowAge' + segmentKey"
                 :id="'crowAgeAdult_' + segmentKey"
                 :value="1"
                 v-model.number="currentLabels.crowAge"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowAgeAdult_' + segmentKey" title="Adult">🧍</label>
      
          <input type="radio" class="btn-check"
                 :name="'crowAge' + segmentKey"
                 :id="'crowAgeJuvenile_' + segmentKey"
                 :value="2"
                 v-model.number="currentLabels.crowAge"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowAgeJuvenile_' + segmentKey" title="Juvenile">👶</label>
        </div>
      
        <!-- Behavior Group -->
        <div class="btn-group btn-group-sm flex-fill mx-1" role="group">
          <input type="checkbox" class="btn-check"
                 :id="'alert_' + segmentKey"
                 v-model="currentLabels.alert"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'alert_' + segmentKey" title="Alert">🚨</label>

          <input type="checkbox" class="btn-check"
                 :id="'mob_' + segmentKey"
                 v-model="currentLabels.mob"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary btn-feature" :for="'mob_' + segmentKey" title="Mob / Attack">⚔️</label>
          
          <input type="checkbox" class="btn-check"
                 :id="'begging_' + segmentKey"
                 v-model="currentLabels.begging"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'begging_' + segmentKey" title="Food / Begging">🥜</label>
          
          <input type="checkbox" class="btn-check"
                 :id="'softSong_' + segmentKey"
                 v-model="currentLabels.softSong"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary btn-feature" :for="'softSong_' + segmentKey" title="Soft Song">🎵</label>

          <input type="checkbox" class="btn-check"
                 :id="'rattle_' + segmentKey"
                 v-model="currentLabels.rattle"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary btn-feature" :for="'rattle_' + segmentKey" title="Rattle">🪇</label>

          <input type="checkbox" class="btn-check"
                 :id="'grief_' + segmentKey"
                 v-model="currentLabels.grief"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'grief_' + segmentKey" title="Grief">💔</label>

        </div>
      
        <!-- Quality Group -->
        <div class="btn-group btn-group-sm flex-fill mx-1" role="group">
          <input type="radio" class="btn-check"
                 :name="'quality' + segmentKey"
                 :id="'quality1_' + segmentKey"
                 :value="1"
                 v-model.number="currentLabels.quality"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-danger btn-feature" :for="'quality1_' + segmentKey" title="Bad Quality">🚫</label>
      
          <input type="radio" class="btn-check"
                 :name="'quality' + segmentKey"
                 :id="'quality2_' + segmentKey"
                 :value="2"
                 v-model.number="currentLabels.quality"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-success" :for="'quality2_' + segmentKey" title="Average Quality">✅</label>
      
          <input type="radio" class="btn-check"
                 :name="'quality' + segmentKey"
                 :id="'quality3_' + segmentKey"
                 :value="3"
                 v-model.number="currentLabels.quality"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'quality3_' + segmentKey" title="Best Quality">⭐</label>
        </div>
      </div>
      
      <!-- Hidden audio element -->
      <audio ref="audio" style="display:none;"></audio>
    </div>
  `,
  data() {
    return {
      isPlaying: false,
      currentTime: this.segment.start_time,
      audio: null
    };
  },
  computed: {
    segmentKey() {
      return `${this.segment.id}-${this.segment.start_time}-${this.segment.end_time}`;
    },
    currentLabels() {
      if (!this.labels[this.segmentKey]) {
        // Initialize with the new structure
        this.labels[this.segmentKey] = {
          crowCount: 1,   // integer [1..4]
          crowAge: 1,     // 1=adult, 2=juvenile
          alert: false,
          begging: false,
          grief: false,
          softSong: false,
          rattle: false,
          mob: false,
          quality: 2,     // integer [1..3]
          reviewed: false
        };
      }
      return this.labels[this.segmentKey];
    },
    cardBorderClass() {
      // Show a primary (blue) border if the segment has been reviewed.
      return this.currentLabels.reviewed
          ? 'border-top border-right border-bottom border-primary'
          : 'border-top border-right border-bottom border-secondary';
    },
    playButtonClass() {
      return this.isPlaying ? 'btn-primary' : 'btn-secondary';
    },
    leftBorderStyle() {
      const colors = ["#007bff", "#28a745", "#ffc107", "#17a2b8", "#6610f2", "#fd7e14", "#6f42c1", "#20c997"];
      let fileId = String(this.segment.id);
      let hash = 0;
      for (let i = 0; i < fileId.length; i++) {
        hash = (hash * 31 + fileId.charCodeAt(i)) % colors.length;
      }
      return { borderLeft: `6px solid ${colors[hash]} !important` };
    }
  },
  methods: {
    formattedTime(seconds) {
      const minutes = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
    },
    handlePlay() {
      // Mark segment as reviewed on play click.
      this.currentLabels.reviewed = true;
      this.onLabelsChanged();
      this.togglePlayback();
    },
    togglePlayback() {
      if (!this.audio) return;
      if (this.isPlaying) {
        this.audio.pause();
        this.isPlaying = false;
      } else {
        if (this.audio.currentTime >= this.segment.end_time) {
          this.audio.currentTime = this.segment.start_time;
          this.currentTime = this.segment.start_time;
        }
        this.audio.play();
        this.isPlaying = true;
      }
    },
    onSliderChange() {
      if (this.audio) {
        this.audio.currentTime = this.currentTime;
      }
    },
    onLabelsChanged() {
      // Whenever any label is changed, mark as reviewed and emit update.
      this.currentLabels.reviewed = true;
      this.$emit('labels-changed', { segmentKey: this.segmentKey, labels: this.currentLabels });
    },
    updateProgress() {
      if (this.audio) {
        this.currentTime = this.audio.currentTime;
        if (this.audio.currentTime >= this.segment.end_time) {
          this.audio.pause();
          this.audio.currentTime = this.segment.start_time;
          this.currentTime = this.segment.start_time;
          this.isPlaying = false;
        }
      }
    },
    copyPrevious() {
      if (this.prevLabels) {
        if (this.audio && this.isPlaying) {
          this.audio.pause();
          this.isPlaying = false;
        }
        // Copy all fields from prevLabels
        const fields = ['crowCount', 'crowAge', 'alert', 'begging', 'grief', 'softSong', 'rattle', 'mob', 'quality', 'notes'];
        fields.forEach(field => {
          this.currentLabels[field] = this.prevLabels[field];
        });
        this.onLabelsChanged();
        this.audio.currentTime = this.segment.start_time;
        this.audio.play();
        this.isPlaying = true;
      }
    }
  },
  mounted() {
    this.audio = this.$refs.audio;
    this.audio.src = `/cache/library/${this.segment.id}.mp3`;
    this.audio.currentTime = this.segment.start_time;
    this.audio.playbackRate = this.playbackSpeed;
    this.audio.addEventListener('timeupdate', this.updateProgress);
  },
  watch: {
    playbackSpeed(newSpeed) {
      if (this.audio) {
        this.audio.playbackRate = newSpeed;
      }
    }
  },
  beforeUnmount() {
    if (this.audio) {
      this.audio.pause();
      this.audio.removeEventListener('timeupdate', this.updateProgress);
      this.audio = null;
    }
  }
};
