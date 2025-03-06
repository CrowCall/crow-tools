// js/components/segmentCard.js
window.SegmentCard = {
  props: ['segment', 'labels', 'playbackSpeed', 'prevLabels'],
  template: `
    <div :class="['card', 'mb-2', 'p-2', cardBorderClass]">
      <!-- 1) Title Row: Recorder + ID, Time, Confidence -->
      <div class="d-flex flex-wrap align-items-center mb-1">
        <strong class="me-2">
          Recorder: {{ segment.recordist }} / ID: {{ segment.id }}
        </strong>
        <small class="text-muted">
          Time: {{ segment.start_time }} - {{ segment.end_time }},
          Conf: {{ segment.confidence.toFixed(2) }}
        </small>
      </div>

      <!-- 2) Notes: single line, read-only, no label -->
      <div class="mb-2" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        <div class="text-muted" :title="segment.media_notes">{{ segment.media_notes }}</div>
      </div>

      <!-- 3) Play button, slider, and copy button -->
      <div class="d-flex align-items-center flex-wrap mb-2" style="gap: 0.5rem;">
        <button :class="['btn', 'btn-sm', playButtonClass]" style="width: 60px;" @click="togglePlayback">
          <span v-if="!isPlaying">Play</span>
          <span v-else>Pause</span>
        </button>
        <input type="range" class="form-range"
               style="width: 120px; margin: 0;"
               :min="segment.start_time"
               :max="segment.end_time"
               :step="0.1"
               v-model.number="currentTime"
               @input="onSliderChange">
        <small>
          {{ formattedTime(currentTime - segment.start_time) }} /
          {{ formattedTime(segment.end_time - segment.start_time) }}
        </small>
        <!-- New Copy Button -->
        <button class="btn btn-sm btn-outline-secondary" style="width: 40px;" @click="copyPrevious" :disabled="!prevLabels">
          📋
        </button>
      </div>

      <!-- 4) Label Buttons -->
      <div class="d-flex flex-wrap align-items-center" style="gap: 0.5rem;">
        <!-- Crow Count Group -->
        <div class="btn-group btn-group-sm" role="group">
          <input type="radio" class="btn-check"
                 :name="'crowCount' + segmentKey"
                 :id="'crowCountSingle_'+segmentKey"
                 value="single"
                 v-model="currentLabels.crowCount"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowCountSingle_'+segmentKey">
            Single
          </label>

          <input type="radio" class="btn-check"
                 :name="'crowCount' + segmentKey"
                 :id="'crowCountMultiple_'+segmentKey"
                 value="multiple"
                 v-model="currentLabels.crowCount"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowCountMultiple_'+segmentKey">
            Multiple
          </label>
        </div>

        <!-- Age Group -->
        <div class="btn-group btn-group-sm" role="group">
          <input type="radio" class="btn-check"
                 :name="'crowAge' + segmentKey"
                 :id="'crowAgeJuvenile_'+segmentKey"
                 value="juvenile"
                 v-model="currentLabels.crowAge"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowAgeJuvenile_'+segmentKey">
            Juvenile
          </label>

          <input type="radio" class="btn-check"
                 :name="'crowAge' + segmentKey"
                 :id="'crowAgeAdult_'+segmentKey"
                 value="adult"
                 v-model="currentLabels.crowAge"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'crowAgeAdult_'+segmentKey">
            Adult
          </label>
        </div>

        <!-- Features Group 1: Beg, Soft, Rattle -->
        <div class="btn-group btn-group-sm" role="group">
          <input type="checkbox" class="btn-check"
                 :id="'begging_'+segmentKey"
                 v-model="currentLabels.begging"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'begging_'+segmentKey">
            Beg
          </label>

          <input type="checkbox" class="btn-check"
                 :id="'softSong_'+segmentKey"
                 v-model="currentLabels.softSong"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'softSong_'+segmentKey">
            Soft
          </label>

          <input type="checkbox" class="btn-check"
                 :id="'rattle_'+segmentKey"
                 v-model="currentLabels.rattle"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-primary" :for="'rattle_'+segmentKey">
            Rattle
          </label>
        </div>

        <!-- Features Group 2: Human and Bad -->
        <div class="btn-group btn-group-sm" role="group">
          <input type="checkbox" class="btn-check"
                 :id="'human_'+segmentKey"
                 v-model="currentLabels.human"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-danger" :for="'human_'+segmentKey">
            Human
          </label>

          <input type="checkbox" class="btn-check"
                 :id="'badQuality_'+segmentKey"
                 v-model="currentLabels.badQuality"
                 @change="onLabelsChanged">
          <label class="btn btn-outline-danger" :for="'badQuality_'+segmentKey">
            Bad
          </label>
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
        this.labels[this.segmentKey] = {
          crowCount: '',
          crowAge: '',
          begging: false,
          softSong: false,
          rattle: false,
          badQuality: false,
          human: false,
          notes: ''
        };
      }
      return this.labels[this.segmentKey];
    },
    cardBorderClass() {
      // Red border if either badQuality or human is true.
      const lbl = this.currentLabels;
      if (lbl.badQuality || lbl.human) {
        return 'border border-danger';
      }
      if (
          lbl.crowCount !== '' ||
          lbl.crowAge !== '' ||
          lbl.begging ||
          lbl.softSong ||
          lbl.rattle ||
          (lbl.notes && lbl.notes.trim() !== '')
      ) {
        return 'border border-primary';
      }
      return 'border border-secondary';
    },
    playButtonClass() {
      return this.isPlaying ? 'btn-primary' : 'btn-secondary';
    }
  },
  methods: {
    formattedTime(seconds) {
      const minutes = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
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
      this.$emit('labels-changed');
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
        // Copy each property from prevLabels into currentLabels, except those we don't want to override.
        const fields = ['crowCount', 'crowAge', 'begging', 'softSong', 'rattle', 'badQuality', 'human'];
        fields.forEach(field => {
          this.currentLabels[field] = this.prevLabels[field];
        });
        this.onLabelsChanged();
      }
    }
  },
  mounted() {
    this.audio = this.$refs.audio;
    this.audio.src = `/library/${this.segment.id}.mp3`;
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
