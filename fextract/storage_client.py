"""First-party storage API client, replacing girder_client for this container's I/O.

Talks to STORAGE_API_URL (retire-girder-dependency Task Group 3.2 — not built as an HTTP surface
yet, so nothing here has been run against a real server) using JOB_AUTH_TOKEN (Task Group 5, a real
per-job-scoped JWT) as Bearer auth. The exact endpoint paths below are a guess at the contract, not
a confirmed API — update once Task Group 3.2 exists for real.

Shared by both CLI entrypoints in this repo (ClassicalFeatures — Extended_Clinical — and
ExpandedGranularFeatures — Feature_Pipeline/Test_Run) and by FeatureExtractor.py, which the latter
instantiates.
"""
import json
import os
import re

import requests

_CONTENT_DISPOSITION_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^;"\n]+)"?')


def _filename_from_response(resp, fallback):
    """Reads the real filename off Content-Disposition rather than the caller guessing an
    extension (e.g. assuming .svs, which is wrong for .tif/.ndpi/etc. slides)."""
    header = resp.headers.get('Content-Disposition', '')
    match = _CONTENT_DISPOSITION_FILENAME_RE.search(header)
    return match.group(1).strip() if match else fallback


class StorageClient:
    def __init__(self, base_url, token):
        self.base_url = base_url.rstrip('/')
        self.token = token

    def _headers(self, **extra):
        return {'Authorization': f'Bearer {self.token}', **extra}

    def _download(self, path, dest_path):
        resp = requests.get(f'{self.base_url}{path}', headers=self._headers(), stream=True)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return resp

    def download_input(self, item_id, dest_dir):
        """Downloads this item's WSI into dest_dir using its real filename (from the server's
        Content-Disposition header), not a guessed extension. Returns the local path."""
        os.makedirs(dest_dir, exist_ok=True)
        tmp_path = os.path.join(dest_dir, f'.{item_id}.download')
        resp = self._download(f'/items/{item_id}/file', tmp_path)
        filename = _filename_from_response(resp, fallback=f'{item_id}.bin')
        dest_path = os.path.join(dest_dir, filename)
        os.replace(tmp_path, dest_path)
        return dest_path

    def get_annotations(self, item_id):
        """Returns this item's annotation layers, newest-updated first. Each entry is
        {"_id": ..., "annotation": {"name": ..., "elements": [...], ...}}, matching the old
        Girder shape this pipeline's own filtering/feature-extraction logic already expects."""
        resp = requests.get(
            f'{self.base_url}/items/{item_id}/annotations', params={'sort': 'updated'}, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def replace_annotation(self, item_id, annotation_doc):
        """Upserts one annotation layer by name — replaces the old delete-then-post pattern
        (Girder's `?token=` query-param auth style dropped along with it)."""
        resp = requests.post(
            f'{self.base_url}/annotation', params={'itemId': item_id}, data=json.dumps(annotation_doc),
            headers=self._headers(**{'Content-Type': 'application/json'}))
        resp.raise_for_status()
        return resp.json()

    def put_item_metadata(self, item_id, metadata: dict):
        resp = requests.put(f'{self.base_url}/items/{item_id}/metadata', json=metadata, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def upload_result_file(self, item_id, filename, local_path):
        with open(local_path, 'rb') as f:
            resp = requests.post(f'{self.base_url}/items/{item_id}/files/{filename}', data=f, headers=self._headers())
        resp.raise_for_status()
        return resp.json()
