// js/components/menu.js
window.MenuComponent = {
  props: {
    title: { type: String, default: 'Crow Tools' }
  },
  template: `
    <nav class="navbar navbar-dark bg-dark mb-4">
      <div class="container-fluid">
        <!-- Brand acts as a refresh on the current page -->
        <a class="navbar-brand fs-2 d-flex align-items-center" href="#" @click.prevent="refreshPage">
          <img src="/images/crow-emote.png" width="40" height="40" class="d-inline-block align-text-top me-2" alt="">
          {{ title }}
        </a>
        <!-- Always-collapsed hamburger -->
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse"
                data-bs-target="#mainNav" aria-controls="mainNav"
                aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="mainNav">
          <ul class="navbar-nav ms-auto mb-2 mb-lg-0">
            <li class="nav-item">
              <a class="nav-link" :class="{ active: isActive('index.html') }" href="index.html">
                Labeler
              </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" :class="{ active: isActive('transcriptions.html') }" href="transcriptions.html">
                Transcriber
              </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" :class="{ active: isActive('embeddings.html') }" href="embeddings.html">
                Embeddings
              </a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  `,
  methods: {
    isActive(page) {
      const current = window.location.pathname.split('/').pop();
      return current === page;
    },
    refreshPage() {
      window.location.reload();
    }
  }
};
