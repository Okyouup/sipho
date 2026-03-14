"""
Aegis-1 — Huawei OBS Memory Sync
==================================
Syncs aegis_memory.json and aegis_knowledge.json
to/from a Huawei Object Storage Service (OBS) bucket.

Install SDK:
    pip install esdk-obs-python

Huawei OBS docs:
    https://support.huaweicloud.com/intl/en-us/sdk-python-devg-obs/obs_22_0001.html

Env vars:
    OBS_ACCESS_KEY   — IAM access key
    OBS_SECRET_KEY   — IAM secret key
    OBS_BUCKET       — bucket name (e.g. aegis-memory-bucket)
    OBS_ENDPOINT     — regional endpoint (e.g. obs.af-south-1.myhuaweicloud.com)
"""
import obs as huawei_obs
import os
import logging
from typing import Optional

log = logging.getLogger("aegis.obs")

# OBS key names inside the bucket
OBS_MEMORY_KEY    = "aegis/aegis_memory.json"
OBS_KNOWLEDGE_KEY = "aegis/aegis_knowledge.json"


class OBSMemorySync:
    """
    Pulls memory files from Huawei OBS on boot.
    Pushes them back on sleep / shutdown.
    This gives Aegis-1 persistent cloud memory that survives
    container restarts and ECS instance replacements.
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        bucket: str,
        endpoint: str,
        local_memory_path: str,
        local_knowledge_path: str,
    ):
        self.bucket              = bucket
        self.local_memory_path   = local_memory_path
        self.local_knowledge_path = local_knowledge_path
        self._client             = None

        try:
            
            self._client = huawei_obs.ObsClient(
                access_key_id=access_key,
                secret_access_key=secret_key,
                server=endpoint,
            )
            log.info(f"OBS client initialised → bucket={bucket} endpoint={endpoint}")
        except ImportError:
            log.warning(
                "esdk-obs-python not installed. OBS sync disabled.\n"
                "Install with: pip install esdk-obs-python"
            )

    def _download(self, obs_key: str, local_path: str) -> bool:
        """Download a single file from OBS to local disk."""
        if self._client is None:
            return False
        try:
            resp = self._client.getObject(self.bucket, obs_key, downloadPath=local_path)
            if resp.status < 300:
                log.info(f"OBS pull: {obs_key} → {local_path}")
                return True
            else:
                # 404 means no memory saved yet — not an error on first run
                if resp.status == 404:
                    log.info(f"OBS: {obs_key} not found (first run) — starting fresh")
                else:
                    log.warning(f"OBS pull failed: {obs_key}  status={resp.status}")
                return False
        except Exception as e:
            log.warning(f"OBS pull error ({obs_key}): {e}")
            return False

    def _upload(self, local_path: str, obs_key: str) -> bool:
        """Upload a single local file to OBS."""
        if self._client is None:
            return False
        if not os.path.exists(local_path):
            log.debug(f"OBS push skipped — {local_path} does not exist yet")
            return False
        try:
            resp = self._client.putFile(self.bucket, obs_key, local_path)
            if resp.status < 300:
                size_kb = os.path.getsize(local_path) / 1024
                log.info(f"OBS push: {local_path} → {obs_key}  ({size_kb:.1f} KB)")
                return True
            else:
                log.warning(f"OBS push failed: {obs_key}  status={resp.status}")
                return False
        except Exception as e:
            log.warning(f"OBS push error ({obs_key}): {e}")
            return False

    def pull(self) -> dict:
        """
        Pull memory and knowledge files from OBS to local disk.
        Call this on boot BEFORE initialising Aegis so it loads
        the cloud-persisted memory.
        """
        log.info("Pulling memory from Huawei OBS...")
        return {
            "memory":    self._download(OBS_MEMORY_KEY,    self.local_memory_path),
            "knowledge": self._download(OBS_KNOWLEDGE_KEY, self.local_knowledge_path),
        }

    def push(self) -> dict:
        """
        Push local memory and knowledge files to OBS.
        Call this after aegis.sleep() so the latest state
        is safely stored in Huawei Cloud.
        """
        log.info("Pushing memory to Huawei OBS...")
        return {
            "memory":    self._upload(self.local_memory_path,    OBS_MEMORY_KEY),
            "knowledge": self._upload(self.local_knowledge_path, OBS_KNOWLEDGE_KEY),
        }

    def status(self) -> dict:
        """Check OBS connectivity and object existence."""
        if self._client is None:
            return {"connected": False, "reason": "SDK not installed"}
        try:
            resp = self._client.headBucket(self.bucket)
            connected = resp.status < 300
            return {
                "connected": connected,
                "bucket": self.bucket,
                "status_code": resp.status,
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}