"""CERN Open Data stream for Z³ training observations.

This module adapts the Cognitive Mesh CERN collision plugin into a standalone,
synchronous stream for the Z3 neural-network runtime. It downloads and caches the
CMS dielectron CSV from CERN Open Data record 304, converts rows into generic
structured observations, and emits batches suitable for `/observe` ingestion.
"""
from __future__ import annotations

import csv
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class CERNCollisionStream:
    """Synchronous CERN CMS dielectron collision-data source."""

    DEFAULT_RECORD_URL = "https://opendata.cern.ch/record/304"
    DEFAULT_DATA_URL = "https://opendata.cern.ch/record/304/files/dielectron.csv?download=1"
    DEFAULT_DOI = "10.7483/OPENDATA.CMS.PCSW.AHVG"
    DEFAULT_DOMAIN = "cern:cms:dielectron"

    def __init__(self) -> None:
        state_dir = Path(os.environ.get("Z3_STATE_DIR", os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "data")))
        self.data_url = os.getenv("CERN_COLLISION_DATA_URL", self.DEFAULT_DATA_URL)
        self.cache_path = Path(os.getenv("CERN_COLLISION_CACHE", str(state_dir / "cern_dielectron.csv")))
        self.batch_size = int(os.getenv("CERN_COLLISION_BATCH_SIZE", "25"))
        self.max_events = int(os.getenv("CERN_COLLISION_MAX_EVENTS", "100000"))
        self.primary_observable = os.getenv("CERN_COLLISION_PRIMARY_VALUE", "M")
        self.secondary_observable = os.getenv("CERN_COLLISION_SECONDARY_VALUE", "pt1")
        self._rows: List[Dict[str, str]] = []
        self._offset = 0
        self._loaded_at: Optional[float] = None
        self._last_batch_at: Optional[float] = None

    def ensure_loaded(self, *, download: bool = True) -> Dict[str, Any]:
        """Ensure the dataset is cached and loaded into memory."""
        if download:
            self.ensure_dataset()
        if not self._rows:
            self.load_rows()
        return self.status()

    def ensure_dataset(self) -> None:
        """Download the CSV if it is not already cached."""
        if self.cache_path.exists() and self.cache_path.stat().st_size > 0:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            self.data_url,
            headers={"User-Agent": "Z3NeuralNetwork-CERNStream/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310 - fixed public HTTPS dataset URL by default.
            content = response.read()
        self.cache_path.write_bytes(content)

    def load_rows(self) -> None:
        """Load cached CSV rows up to max_events."""
        if not self.cache_path.exists():
            raise FileNotFoundError(f"CERN cache not found: {self.cache_path}")
        with self.cache_path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            rows: List[Dict[str, str]] = []
            for idx, row in enumerate(reader):
                if idx >= self.max_events:
                    break
                if row:
                    rows.append(row)
        self._rows = rows
        self._loaded_at = time.time()
        self._offset = min(self._offset, max(len(self._rows) - 1, 0))

    def fetch_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Return a batch of converted observations and advance the stream offset."""
        self.ensure_loaded(download=True)
        count = int(batch_size or self.batch_size)
        observations: List[Dict[str, Any]] = []
        if not self._rows:
            return {"domain": self.DEFAULT_DOMAIN, "observations": observations, "offset": self._offset, "count": 0}
        for _ in range(min(count, len(self._rows))):
            row = self._rows[self._offset]
            self._offset = (self._offset + 1) % len(self._rows)
            obs = self.row_to_observation(row)
            if obs is not None:
                observations.append(obs)
        self._last_batch_at = time.time()
        return {
            "domain": self.DEFAULT_DOMAIN,
            "observations": observations,
            "offset": self._offset,
            "count": len(observations),
            "dataset": self.dataset_info(),
        }

    def dataset_info(self) -> Dict[str, Any]:
        return {
            "source": "CERN Open Data Portal",
            "record_url": self.DEFAULT_RECORD_URL,
            "data_url": self.data_url,
            "doi": self.DEFAULT_DOI,
            "experiment": "CMS",
            "accelerator": "CERN-LHC",
            "dataset": "Events with two electrons from 2010",
            "event_type": "dielectron",
            "domain": self.DEFAULT_DOMAIN,
            "primary_observable": self.primary_observable,
            "secondary_observable": self.secondary_observable,
            "unit": "GeV",
        }

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": bool(self._rows),
            "rows_loaded": len(self._rows),
            "offset": self._offset,
            "cache_path": str(self.cache_path),
            "cache_exists": self.cache_path.exists(),
            "cache_size_bytes": self.cache_path.stat().st_size if self.cache_path.exists() else 0,
            "loaded_at": self._loaded_at,
            "last_batch_at": self._last_batch_at,
            "batch_size": self.batch_size,
            "max_events": self.max_events,
            "dataset": self.dataset_info(),
        }

    def row_to_observation(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Translate a CMS dielectron CSV row into a generic Z3 observation."""
        try:
            invariant_mass = self._field_float(row, "M")
            electron_energy_1 = self._field_float(row, "E1", "E")
            electron_energy_2 = self._field_float(row, "E2")
            transverse_momentum_1 = self._field_float(row, "pt1", "pt")
            transverse_momentum_2 = self._field_float(row, "pt2")
            pseudorapidity_1 = self._field_float(row, "eta1", "eta")
            pseudorapidity_2 = self._field_float(row, "eta2")
            phi_1 = self._field_float(row, "phi1", "phi")
            phi_2 = self._field_float(row, "phi2")
            charge_1 = self._field_float(row, "Q1", "Q")
            charge_2 = self._field_float(row, "Q2")
            px_1 = self._field_float(row, "px1", "px")
            py_1 = self._field_float(row, "py1", "py")
            pz_1 = self._field_float(row, "pz1", "pz")
            px_2 = self._field_float(row, "px2")
            py_2 = self._field_float(row, "py2")
            pz_2 = self._field_float(row, "pz2")

            value = self._field_float(row, self.primary_observable)
            if value is None:
                value = invariant_mass if invariant_mass is not None else electron_energy_1
            if value is None:
                return None
            secondary_value = self._field_float(row, self.secondary_observable)
            if secondary_value is None:
                pts = [v for v in (transverse_momentum_1, transverse_momentum_2) if v is not None]
                secondary_value = sum(pts) / len(pts) if pts else None

            run = (row.get("Run") or "unknown").strip()
            event = (row.get("Event") or str(self._offset)).strip()
            entity_id = f"cms_dielectron_run{run}_event{event}"

            momentum_norm_1 = None
            if px_1 is not None and py_1 is not None and pz_1 is not None:
                momentum_norm_1 = math.sqrt(px_1 * px_1 + py_1 * py_1 + pz_1 * pz_1)
            momentum_norm_2 = None
            if px_2 is not None and py_2 is not None and pz_2 is not None:
                momentum_norm_2 = math.sqrt(px_2 * px_2 + py_2 * py_2 + pz_2 * pz_2)

            return {
                "entity_id": entity_id,
                "symbol": entity_id,
                "domain_prefix": "cern",
                "concept": "cms_dielectron_collision",
                "value": float(value),
                "secondary_value": float(secondary_value) if secondary_value is not None else 0.0,
                "timestamp": time.time(),
                **self.dataset_info(),
                "run": run,
                "event": event,
                "invariant_mass_gev": invariant_mass,
                "electron_1_energy_gev": electron_energy_1,
                "electron_2_energy_gev": electron_energy_2,
                "electron_1_transverse_momentum_gev": transverse_momentum_1,
                "electron_2_transverse_momentum_gev": transverse_momentum_2,
                "electron_1_pseudorapidity_eta": pseudorapidity_1,
                "electron_2_pseudorapidity_eta": pseudorapidity_2,
                "electron_1_phi_rad": phi_1,
                "electron_2_phi_rad": phi_2,
                "electron_1_charge": charge_1,
                "electron_2_charge": charge_2,
                "electron_1_momentum_norm_gev": momentum_norm_1,
                "electron_2_momentum_norm_gev": momentum_norm_2,
            }
        except Exception:
            return None

    @classmethod
    def _field_float(cls, row: Dict[str, str], *names: str) -> Optional[float]:
        normalized = {str(key).strip(): value for key, value in row.items()}
        for name in names:
            parsed = cls._float(normalized.get(str(name).strip()))
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            text = str(value).strip()
            if not text:
                return None
            return float(text)
        except (TypeError, ValueError):
            return None
