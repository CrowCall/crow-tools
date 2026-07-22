import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from downloader import download_crows_xeno as xc


class FakeResp:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload


RECORDINGS = [
    {"id": "160878", "date": "2013-01-01", "lat": "47.6", "lng": "-122.3",
     "rec": "A. Recordist", "rmk": "flight call", "file": "//xeno-canto.org/160878/download"},
    {"id": "163637", "date": "2014-05-02", "lat": "48.1", "lng": "-121.9",
     "rec": "B. Recordist", "rmk": "", "file": {"url": "https://xeno-canto.org/163637/download"}},
    {"id": "999999", "date": "bad-date", "file": "//x/999999/download"},
]


def test_fetch_hits_v3_endpoint_with_key_and_tag_query(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"] = url
        seen["params"] = params
        return FakeResp({"numPages": 1, "recordings": RECORDINGS})

    monkeypatch.setattr(xc.requests, "get", fake_get)
    recs = xc.fetch_all_recordings(xc.SPECIES_QUERY, "TESTKEY")

    assert seen["url"] == "https://xeno-canto.org/api/3/recordings"
    assert seen["params"]["key"] == "TESTKEY"
    assert seen["params"]["query"] == xc.SPECIES_QUERY
    assert len(recs) == 3


def test_recording_file_url_handles_string_dict_and_scheme():
    assert xc.recording_file_url({"file": "//x/1/download"}) == "https://x/1/download"
    assert xc.recording_file_url({"file": {"url": "https://x/2/download"}}) == "https://x/2/download"
    assert xc.recording_file_url({"file": "https://x/3/download"}) == "https://x/3/download"
    assert xc.recording_file_url({}) == ""


def test_missing_key_skips_without_network(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("XENO_CANTO_API_KEY", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("network must not be touched without a key")

    monkeypatch.setattr(xc.requests, "get", boom)
    xc.start_downloads(cache_base=str(tmp_path))

    out = capsys.readouterr().out
    assert "requires a free key" in out
    assert "xeno-canto.org/account" in out


def test_start_downloads_writes_catalog_and_applies_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(xc, "fetch_all_recordings", lambda query, key: RECORDINGS)
    downloaded = []
    monkeypatch.setattr(xc, "download_mp3",
                        lambda xc_id, url, out: downloaded.append((xc_id, url)))

    xc.start_downloads(selected_ids=["160878", "163637"],
                       cache_base=str(tmp_path), api_key="TESTKEY")

    csv_path = tmp_path / "libraries" / "xeno-canto" / "library.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    ids = {r["ML Catalog Number"] for r in rows}

    assert ids == {"160878", "163637"}
    assert dict(downloaded) == {
        "160878": "https://xeno-canto.org/160878/download",
        "163637": "https://xeno-canto.org/163637/download",
    }
    notes = {r["ML Catalog Number"]: r["Media notes"] for r in rows}
    assert notes["160878"] == "flight call"
