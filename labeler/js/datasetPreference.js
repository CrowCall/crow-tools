window.DatasetPreference = {
  cookieName: 'crowtools_dataset',
  defaultDataset: 'all-public',

  getCookie(name) {
    const prefix = `${encodeURIComponent(name)}=`;
    const cookies = document.cookie ? document.cookie.split('; ') : [];
    for (const cookie of cookies) {
      if (cookie.startsWith(prefix)) {
        return decodeURIComponent(cookie.slice(prefix.length));
      }
    }
    return null;
  },

  setCookie(name, value) {
    const maxAge = 60 * 60 * 24 * 365;
    document.cookie = [
      `${encodeURIComponent(name)}=${encodeURIComponent(value)}`,
      'path=/',
      `max-age=${maxAge}`,
      'samesite=lax'
    ].join('; ');
  },

  resolveDataset() {
    const params = new URLSearchParams(window.location.search);
    const fromUrl = params.get('dataset');
    if (fromUrl) {
      this.setCookie(this.cookieName, fromUrl);
      return fromUrl;
    }

    return this.getCookie(this.cookieName) || this.defaultDataset;
  },

  persistDataset(dataset) {
    if (dataset) {
      this.setCookie(this.cookieName, dataset);
    }
  },

  withDataset(url, dataset) {
    const nextUrl = new URL(url, window.location.href);
    if (dataset) {
      nextUrl.searchParams.set('dataset', dataset);
    }
    return `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
  }
};
