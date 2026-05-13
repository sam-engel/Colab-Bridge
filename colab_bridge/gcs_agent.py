"""
GCS Agent for Colab Bridge
==========================
Wraps google-cloud-storage with parallel upload/download and billing tracking.

Recommended init for bridge-submitted jobs::

    from colab_bridge.gcs_agent import GCSAgent
    gcs = GCSAgent.from_env('my-bucket')             # reads os.environ['gcs_creds']
    gcs.upload('local/dir/', 'remote/prefix/')
    gcs.download('remote/prefix/', 'local/dir/')

Why `from_env`: the colab bridge auto-injects every key in `colab_bridge/.env`
into the runtime as an environment variable before your code runs. The Colab
Secrets API (`from google.colab import userdata`) does NOT work from the bridge
— it times out with "Secrets can only be fetched when running from the Colab UI."
`from_colab_secrets()` is kept only for backward compat (it falls back to
`from_env`). Use `from_env` directly in any new script.

To set the credential JSON once::

    colab --env-set gcs_creds --from-file path/to/service-account.json
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def _ensure_gcs():
    try:
        from google.cloud import storage  # noqa: F401
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', 'google-cloud-storage'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


class GCSAgent:
    """
    Google Cloud Storage agent with billing tracking.

    Class A ops (Insert/Update/List): $0.05 / 10,000
    Class B ops (Get):                $0.004 / 10,000
    Free ops   (Delete):              $0.00
    """

    def __init__(self, bucket_name=None, creds='gc_credentials.json',
                 verbose=True, max_workers=10):
        _ensure_gcs()
        from google.cloud import storage

        self.verbose = verbose
        self.max_workers = max_workers
        self.stats = {'class_a': 0, 'class_b': 0, 'free_ops': 0}

        if isinstance(creds, dict):
            import tempfile, json as _json
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                _json.dump(creds, f)
                creds = f.name

        if not os.path.exists(creds):
            raise FileNotFoundError(f'Credentials not found: {creds}')

        self.storage_client = storage.Client.from_service_account_json(creds)

        if bucket_name is None:
            self.stats['class_a'] += 1
            self.bucket = next(self.storage_client.list_buckets(), None)
            if self.bucket is None:
                raise ValueError('No buckets found in this project.')
            if self.verbose:
                print(f'GCSAgent: connected to bucket "{self.bucket.name}"')
        else:
            self.bucket = self.storage_client.bucket(bucket_name)
            if self.verbose:
                print(f'GCSAgent: using bucket "{bucket_name}"')

    @classmethod
    def from_colab_secrets(cls, bucket_name=None, **kwargs):
        """DEPRECATED: Colab Secrets are unreachable from the bridge. Falls back
        to from_env() which reads `os.environ['gcs_creds']` (auto-injected by
        the bridge from `colab_bridge/.env`). Kept only for back-compat with
        old code (e.g. exp06).
        """
        try:
            return cls.from_env(bucket_name=bucket_name, **kwargs)
        except Exception:
            try:
                from google.colab import userdata
                creds_str = userdata.get('gcs_creds')
            except Exception as e:
                raise RuntimeError(
                    f'gcs_creds unavailable: env var not set and userdata unreachable ({e}). '
                    f'Set with: colab --env-set gcs_creds --from-file path/to/key.json'
                )
            creds_dict = json.loads(creds_str)
            return cls(bucket_name=bucket_name, creds=creds_dict, **kwargs)

    @classmethod
    def from_env(cls, bucket_name=None, env_var='gcs_creds', **kwargs):
        """Initialise from `os.environ[env_var]` (JSON string).

        Use this from bridge-submitted jobs — the bridge auto-injects every key
        in `colab_bridge/.env` as an environment variable before your code runs.
        Set the credential JSON once with::

            colab --env-set gcs_creds --from-file path/to/service-account.json
        """
        creds_str = os.environ.get(env_var)
        if not creds_str:
            raise RuntimeError(
                f'Env var {env_var!r} is not set. The bridge injects it from '
                f'`colab_bridge/.env`. Set with: '
                f'colab --env-set {env_var} --from-file path/to/key.json'
            )
        creds_dict = json.loads(creds_str)
        return cls(bucket_name=bucket_name, creds=creds_dict, **kwargs)

    # ── cost tracking ──────────────────────────────────────────────────────────

    def print_usage(self):
        print('\n--- GCS API Usage ---')
        print(f'Class A (Write/List): {self.stats["class_a"]}')
        print(f'Class B (Read):       {self.stats["class_b"]}')
        print(f'Free (Delete):        {self.stats["free_ops"]}')
        rate_a = 0.05 / 10_000
        rate_b = 0.004 / 10_000
        cost = self.stats['class_a'] * rate_a + self.stats['class_b'] * rate_b
        print(f'Est. cost: ${cost:.6f}')

    # ── public API ─────────────────────────────────────────────────────────────

    def upload(self, src, dst):
        """Upload local file or directory to GCS prefix."""
        src_files, root_path = self._resolve_local_source(src)
        if not src_files:
            return
        dst_blobs = self._resolve_remote_dest(src_files, root_path, dst)
        self._run_parallel(self._upload_single, list(zip(src_files, dst_blobs)), 'Uploading')

    def download(self, src, dst):
        """Download GCS prefix to local directory."""
        src_blobs = self._resolve_remote_source(src)
        if not src_blobs:
            if self.verbose:
                print('GCSAgent: source not found:', src)
            return
        dst_files = self._resolve_local_dest(src, src_blobs, dst)
        self._run_parallel(self._download_single, list(zip(src_blobs, dst_files)), 'Downloading')

    def ls(self, prefix=None, recursive=False, max_results=1000):
        """List blobs under a prefix. Returns list of blob names."""
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        delimiter = None if recursive else '/'
        self.stats['class_a'] += 1
        iterator = self.storage_client.list_blobs(
            self.bucket, prefix=prefix, delimiter=delimiter, max_results=max_results
        )
        results = []
        for blob in iterator:
            results.append(blob.name)
            if max_results and len(results) >= max_results:
                return results
        if delimiter and iterator.prefixes:
            for folder in iterator.prefixes:
                if max_results and len(results) >= max_results:
                    break
                results.append(folder)
        return results

    def delete(self, items, silent=False):
        blob_names = self._resolve_remote_source(items)
        if not blob_names:
            return
        batches = [blob_names[i:i + 100] for i in range(0, len(blob_names), 100)]
        it = tqdm(batches, desc='Deleting', disable=not self.verbose) if not silent else batches
        for batch in it:
            with self.storage_client.batch():
                for name in batch:
                    self.bucket.blob(name).delete()
                    self.stats['free_ops'] += 1

    # ── internals ─────────────────────────────────────────────────────────────

    def _run_parallel(self, task_func, items, desc):
        if not items:
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(task_func, *item) for item in items]
            for _ in tqdm(as_completed(futures), total=len(futures),
                          desc=desc, disable=not self.verbose):
                pass
            for f in futures:
                f.result()

    def _upload_single(self, local, remote):
        self.stats['class_a'] += 1
        self.bucket.blob(remote).upload_from_filename(local)

    def _download_single(self, remote, local):
        self.stats['class_b'] += 1
        os.makedirs(os.path.dirname(local), exist_ok=True)
        self.bucket.blob(remote).download_to_filename(local)

    def _resolve_local_source(self, src):
        files, root = [], ''
        if isinstance(src, list):
            return src, None
        if os.path.isdir(src):
            root = src
            for dp, _, fns in os.walk(src):
                for fn in fns:
                    files.append(os.path.join(dp, fn))
        else:
            files, root = [src], os.path.dirname(src)
        return files, root

    def _resolve_remote_source(self, src):
        if isinstance(src, list):
            return src
        return self.ls(prefix=src, recursive=True, max_results=None)

    def _resolve_remote_dest(self, src_files, root_path, dst):
        results = []
        if isinstance(dst, list):
            return dst
        for local in src_files:
            if root_path:
                rel = os.path.relpath(local, root_path).replace(os.sep, '/')
                results.append(os.path.join(dst, rel).replace('\\', '/'))
            else:
                results.append(
                    os.path.join(dst, os.path.basename(local)).replace('\\', '/'))
        return results

    def _resolve_local_dest(self, raw_src, src_blobs, dst):
        results = []
        dst = os.path.normpath(dst)
        for blob_name in src_blobs:
            if isinstance(raw_src, str) and raw_src.endswith('/'):
                rel = blob_name[len(raw_src):]
                if not rel:
                    continue
                results.append(os.path.join(dst, rel))
            else:
                results.append(os.path.join(dst, os.path.basename(blob_name)))
        return results
