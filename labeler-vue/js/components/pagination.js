// js/components/pagination.js
window.Pagination = {
    props: ['currentPage', 'totalPages'],
    data() {
        return {
            pageSelection: this.currentPage
        };
    },
    watch: {
        currentPage(newVal) {
            this.pageSelection = newVal;
        }
    },
    template: `
      <div class="mb-3 d-flex flex-wrap align-items-center text-nowrap" style="gap: 0.5rem; color: #fff;">
        <button class="btn btn-primary" @click="$emit('prev')" :disabled="currentPage === 1">
          Previous
        </button>
        <button class="btn btn-primary" @click="$emit('next')" :disabled="currentPage === totalPages">
          Next
        </button>
        <span>Page</span>
        <select class="form-select form-select-sm" style="width: auto;"
                v-model.number="pageSelection" @change="goToPage">
          <option v-for="p in totalPages" :value="p" :key="p">{{ p }}</option>
        </select>
      </div>
    `,
    methods: {
        goToPage() {
            this.$emit('goto', this.pageSelection);
        }
    }
};
