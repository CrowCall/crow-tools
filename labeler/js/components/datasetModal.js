window.DatasetModal = {
  props: ['currentDataset'],
  template: `
    <div class="ms-auto me-2">
      <span class="badge bg-info text-dark fs-5 px-3 py-2" role="button" data-bs-toggle="modal" data-bs-target="#datasetModal" @click="openModal">{{ label }}</span>
      <div class="modal fade" id="datasetModal" tabindex="-1" aria-labelledby="datasetModalLabel" aria-hidden="true">
        <div class="modal-dialog">
          <div class="modal-content">
            <div class="modal-header">
              <h5 class="modal-title" id="datasetModalLabel">Datasets</h5>
              <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
              <div class="mb-3">
                <label class="form-label">Select Dataset</label>
                <select class="form-select" v-model="selected">
                  <option v-for="d in datasets" :key="d" :value="d">{{ d }}</option>
                </select>
                <button class="btn btn-primary mt-2" @click="chooseDataset">Use Dataset</button>
              </div>
              <hr/>
              <div>
                <h6>Create / Edit Dataset</h6>
                <div class="mb-2">
                  <input type="text" class="form-control" placeholder="Dataset name" v-model="form.name" />
                </div>
                <div class="mb-2">
                  <label class="form-label">Libraries</label>
                  <div class="form-check" v-for="lib in libraries" :key="lib">
                    <input class="form-check-input" type="checkbox" :id="'lib_'+lib" v-model="form.libs[lib]">
                    <label class="form-check-label" :for="'lib_'+lib">{{ lib }}</label>
                  </div>
                </div>
                <div class="mb-2">
                  <label class="form-label">Import labels from</label>
                  <select class="form-select" v-model="form.importFrom">
                    <option value="">None</option>
                    <option v-for="d in datasets" :key="'imp_'+d" :value="d">{{ d }}</option>
                  </select>
                </div>
                <button class="btn btn-success" @click="saveDataset">Save</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  data() {
    return {
      datasets: [],
      libraries: [],
      selected: this.currentDataset,
      form: {name:'', libs:{}, importFrom:''}
    };
  },
  computed: {
    label() { return this.selected || 'Dataset'; }
  },
  methods: {
    openModal() {
      fetch('/datasets').then(r=>r.json()).then(data=>{ this.datasets = data.datasets; });
      fetch('/libraries')
        .then(r=>r.json())
        .then(data => {
          this.libraries = data.libraries || [];
          if(!this.form.name) this.form.libs = {};
          this.libraries.forEach(l => { if(!(l in this.form.libs)) this.form.libs[l] = false; });
        })
        .catch(()=>{});
    },
    chooseDataset(){ this.$emit('dataset-changed', this.selected); },
    saveDataset(){
      const libs = Object.keys(this.form.libs).filter(l => this.form.libs[l]);
      if(!this.form.name) return;
      const method = this.datasets.includes(this.form.name) ? 'PUT' : 'POST';
      const url = method === 'POST' ? '/datasets' : `/datasets/${encodeURIComponent(this.form.name)}`;
      fetch(url, {
        method,
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({name:this.form.name, included_libraries: libs, importFrom: this.form.importFrom})
      }).then(() => {
        this.selected = this.form.name;
        this.$emit('dataset-changed', this.selected);
        const el = document.getElementById('datasetModal');
        if(el){
          const modal = bootstrap.Modal.getInstance(el) || new bootstrap.Modal(el);
          modal.hide();
        }
        this.form = {name:'', libs:{}, importFrom:''};
      });
    }
  },
  watch:{
    currentDataset(val){ this.selected=val; }
  }
};
