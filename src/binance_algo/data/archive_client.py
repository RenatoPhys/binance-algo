"""Resumable, checksummed downloader for official Binance public archives."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import os
import random
import re
import stat
import time
import uuid
import zipfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath

import aiohttp

from binance_algo.common.errors import ArchiveError, InvalidStateTransition, StateStoreError
from binance_algo.data.manifest import (
    DataFileRecord,
    DataFileStatus,
    deterministic_file_id,
    now_ms,
)
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage

KLINE_HEADER = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{3,30}$")
INTERVAL_PATTERN = re.compile(r"^[0-9]+[smhdw]$")
CHECKSUM_PATTERN = re.compile(r"^([0-9a-fA-F]{64})\s+\*?([^\s]+)$")
Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class ArchiveTarget:
    symbol: str
    interval: str
    day: date
    dataset: str = "klines"

    def __post_init__(self) -> None:
        if not SYMBOL_PATTERN.fullmatch(self.symbol):
            raise ArchiveError(f"invalid archive symbol: {self.symbol!r}")
        if not INTERVAL_PATTERN.fullmatch(self.interval):
            raise ArchiveError(f"invalid archive interval: {self.interval!r}")
        if self.dataset != "klines":
            raise ArchiveError(f"unsupported archive dataset: {self.dataset}")

    @property
    def filename(self) -> str:
        return f"{self.symbol}-{self.interval}-{self.day.isoformat()}.zip"

    @property
    def csv_filename(self) -> str:
        return f"{self.symbol}-{self.interval}-{self.day.isoformat()}.csv"

    @property
    def start_time_ms(self) -> int:
        return int(datetime.combine(self.day, datetime.min.time(), tzinfo=UTC).timestamp() * 1_000)

    @property
    def end_time_ms(self) -> int:
        return self.start_time_ms + int(timedelta(days=1).total_seconds() * 1_000) - 1

    def url(self, base_url: str) -> str:
        return (
            f"{base_url.rstrip('/')}/futures/um/daily/{self.dataset}/"
            f"{self.symbol}/{self.interval}/{self.filename}"
        )


class DownloadOutcome(StrEnum):
    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArchiveDownloadResult:
    target: ArchiveTarget
    outcome: DownloadOutcome
    archive_path: str
    extracted_path: str | None
    checksum: str | None
    row_count: int | None
    bytes_downloaded: int
    duration_ms: float
    resumed: bool
    error: str | None = None


def parse_checksum(content: str, *, expected_filename: str) -> str:
    match = CHECKSUM_PATTERN.fullmatch(content.strip())
    if match is None:
        raise ArchiveError("invalid Binance CHECKSUM format")
    checksum, filename = match.groups()
    if filename != expected_filename:
        raise ArchiveError(
            f"CHECKSUM filename mismatch: expected {expected_filename}, received {filename}"
        )
    return checksum.lower()


def sha256_file(path: Path, *, chunk_bytes: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise ArchiveError(f"cannot hash archive {path}: {exc}") from exc
    return digest.hexdigest()


def validate_kline_csv(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if tuple(header or ()) != KLINE_HEADER:
                raise ArchiveError(f"unexpected kline CSV header in {path}")
            row_count = 0
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(KLINE_HEADER):
                    raise ArchiveError(
                        f"kline CSV row {line_number} has {len(row)} columns, expected 12"
                    )
                try:
                    int(row[0])
                    int(row[6])
                    int(row[8])
                except ValueError as exc:
                    raise ArchiveError(
                        f"kline CSV row {line_number} has invalid integer fields"
                    ) from exc
                row_count += 1
    except (OSError, UnicodeError) as exc:
        raise ArchiveError(f"cannot validate kline CSV {path}: {exc}") from exc
    if row_count == 0:
        raise ArchiveError(f"kline CSV has no data rows: {path}")
    return row_count


def validate_and_extract_kline_archive(
    archive_path: Path,
    extracted_path: Path,
    *,
    expected_csv_filename: str,
    max_uncompressed_bytes: int,
    chunk_bytes: int,
) -> int:
    """Validate a flat, single-file ZIP and atomically extract its canonical CSV."""

    if extracted_path.exists():
        return validate_kline_csv(extracted_path)

    temporary = extracted_path.with_name(f".{extracted_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise ArchiveError(
                    f"archive must contain exactly one CSV, found {len(members)} entries"
                )
            member = members[0]
            member_path = PurePosixPath(member.filename)
            if (
                member.is_dir()
                or member_path.is_absolute()
                or ".." in member_path.parts
                or len(member_path.parts) != 1
                or member_path.name != expected_csv_filename
            ):
                raise ArchiveError(f"unsafe or unexpected ZIP member: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ArchiveError(f"ZIP symlink is prohibited: {member.filename}")
            if member.file_size <= 0 or member.file_size > max_uncompressed_bytes:
                raise ArchiveError(
                    f"uncompressed archive size {member.file_size} is outside configured bounds"
                )

            extracted_path.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with archive.open(member, "r") as source, temporary.open("xb") as destination:
                while chunk := source.read(chunk_bytes):
                    total += len(chunk)
                    if total > max_uncompressed_bytes:
                        raise ArchiveError(
                            "archive exceeded uncompressed size limit while extracting"
                        )
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        row_count = validate_kline_csv(temporary)
        os.replace(temporary, extracted_path)
        return row_count
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArchiveError(f"cannot validate/extract archive {archive_path}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


class ArchiveDownloader:
    def __init__(
        self,
        *,
        base_url: str,
        storage: LocalFilesystemStorage,
        state_store: StateStore,
        request_timeout_seconds: float,
        max_concurrency: int,
        max_attempts: int,
        retry_base_seconds: float,
        max_archive_bytes: int,
        max_uncompressed_bytes: int,
        chunk_bytes: int,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._storage = storage
        self._state_store = state_store
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._max_archive_bytes = max_archive_bytes
        self._max_uncompressed_bytes = max_uncompressed_bytes
        self._chunk_bytes = chunk_bytes
        self._sleep = sleep
        self._jitter: Jitter = jitter or (lambda ceiling: random.uniform(0.0, ceiling))

    async def download_many(
        self, targets: Sequence[ArchiveTarget], *, ingestion_run_id: str
    ) -> list[ArchiveDownloadResult]:
        async with aiohttp.ClientSession(
            headers={"User-Agent": "binance-algo/0.2 public-archive"}
        ) as session:
            tasks = [
                self._guarded_download(session, target, ingestion_run_id=ingestion_run_id)
                for target in targets
            ]
            return list(await asyncio.gather(*tasks))

    async def _guarded_download(
        self,
        session: aiohttp.ClientSession,
        target: ArchiveTarget,
        *,
        ingestion_run_id: str,
    ) -> ArchiveDownloadResult:
        async with self._semaphore:
            started = time.monotonic_ns()
            archive_path = self._archive_path(target)
            file_id = self._file_id(target)
            try:
                return await self._download_one(
                    session,
                    target,
                    ingestion_run_id=ingestion_run_id,
                    started_ns=started,
                )
            except (ArchiveError, StateStoreError, OSError) as exc:
                existing = self._state_store.get_data_file(file_id)
                if existing is not None and existing.status not in {
                    DataFileStatus.FAILED,
                    DataFileStatus.QUARANTINED,
                }:
                    with suppress(InvalidStateTransition):
                        self._state_store.transition_data_file(
                            file_id, DataFileStatus.FAILED, last_error=str(exc)
                        )
                return ArchiveDownloadResult(
                    target=target,
                    outcome=DownloadOutcome.FAILED,
                    archive_path=str(archive_path),
                    extracted_path=None,
                    checksum=None,
                    row_count=None,
                    bytes_downloaded=0,
                    duration_ms=(time.monotonic_ns() - started) / 1_000_000,
                    resumed=False,
                    error=str(exc),
                )

    async def _download_one(
        self,
        session: aiohttp.ClientSession,
        target: ArchiveTarget,
        *,
        ingestion_run_id: str,
        started_ns: int,
    ) -> ArchiveDownloadResult:
        archive_path = self._archive_path(target)
        checksum_path = archive_path.with_name(f"{archive_path.name}.CHECKSUM")
        extracted_path = archive_path.parent / "extracted" / target.csv_filename
        file_id = self._file_id(target)
        timestamp = now_ms()
        record = self._state_store.register_data_file(
            DataFileRecord(
                file_id=file_id,
                logical_dataset="klines",
                layer="raw_archives",
                source="binance_public_data",
                symbol=target.symbol,
                interval=target.interval,
                start_time_ms=target.start_time_ms,
                end_time_ms=target.end_time_ms,
                row_count=None,
                schema_version=1,
                checksum=None,
                path=str(archive_path),
                status=DataFileStatus.DOWNLOADING,
                created_at_ms=timestamp,
                updated_at_ms=timestamp,
                ingestion_run_id=ingestion_run_id,
            )
        )

        try:
            local = self._validate_local(
                target,
                archive_path=archive_path,
                checksum_path=checksum_path,
                extracted_path=extracted_path,
            )
        except ArchiveError:
            local = None
        if local is not None:
            checksum, row_count = local
            self._promote_manifest_to_validated(record, checksum=checksum, row_count=row_count)
            return ArchiveDownloadResult(
                target=target,
                outcome=DownloadOutcome.SKIPPED,
                archive_path=str(archive_path),
                extracted_path=str(extracted_path),
                checksum=checksum,
                row_count=row_count,
                bytes_downloaded=0,
                duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
                resumed=False,
            )

        invalid_local_files = any(
            path.exists() for path in (archive_path, checksum_path, extracted_path)
        )
        if invalid_local_files:
            self._quarantine_existing(target, archive_path, checksum_path, extracted_path)
            if record.status not in {
                DataFileStatus.FAILED,
                DataFileStatus.QUARANTINED,
            }:
                record = self._state_store.transition_data_file(
                    file_id, DataFileStatus.QUARANTINED, last_error="local validation failed"
                )
            record = self._state_store.transition_data_file(file_id, DataFileStatus.DOWNLOADING)
        elif record.status in {DataFileStatus.FAILED, DataFileStatus.QUARANTINED}:
            record = self._state_store.transition_data_file(file_id, DataFileStatus.DOWNLOADING)

        checksum_content = await self._download_small(
            session, f"{target.url(self._base_url)}.CHECKSUM", max_bytes=1_024
        )
        try:
            checksum_text = checksum_content.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ArchiveError("Binance CHECKSUM response is not ASCII") from exc
        expected_checksum = parse_checksum(checksum_text, expected_filename=target.filename)
        self._write_checksum(checksum_path, checksum_content)

        bytes_downloaded, resumed = await self._download_archive(
            session,
            target.url(self._base_url),
            archive_path=archive_path,
            expected_checksum=expected_checksum,
        )
        self._state_store.transition_data_file(
            file_id, DataFileStatus.DOWNLOADED, checksum=expected_checksum
        )
        row_count = validate_and_extract_kline_archive(
            archive_path,
            extracted_path,
            expected_csv_filename=target.csv_filename,
            max_uncompressed_bytes=self._max_uncompressed_bytes,
            chunk_bytes=self._chunk_bytes,
        )
        self._state_store.transition_data_file(
            file_id,
            DataFileStatus.VALIDATED,
            checksum=expected_checksum,
            row_count=row_count,
        )
        return ArchiveDownloadResult(
            target=target,
            outcome=DownloadOutcome.DOWNLOADED,
            archive_path=str(archive_path),
            extracted_path=str(extracted_path),
            checksum=expected_checksum,
            row_count=row_count,
            bytes_downloaded=bytes_downloaded,
            duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            resumed=resumed,
        )

    def _validate_local(
        self,
        target: ArchiveTarget,
        *,
        archive_path: Path,
        checksum_path: Path,
        extracted_path: Path,
    ) -> tuple[str, int] | None:
        if not archive_path.exists() or not checksum_path.exists():
            return None
        try:
            checksum = parse_checksum(
                checksum_path.read_text(encoding="ascii"), expected_filename=target.filename
            )
        except (OSError, UnicodeError) as exc:
            raise ArchiveError(f"cannot read local checksum {checksum_path}: {exc}") from exc
        if sha256_file(archive_path, chunk_bytes=self._chunk_bytes) != checksum:
            return None
        row_count = validate_and_extract_kline_archive(
            archive_path,
            extracted_path,
            expected_csv_filename=target.csv_filename,
            max_uncompressed_bytes=self._max_uncompressed_bytes,
            chunk_bytes=self._chunk_bytes,
        )
        return checksum, row_count

    def _promote_manifest_to_validated(
        self, record: DataFileRecord, *, checksum: str, row_count: int
    ) -> None:
        current = record
        if current.status in {DataFileStatus.FAILED, DataFileStatus.QUARANTINED}:
            current = self._state_store.transition_data_file(
                current.file_id, DataFileStatus.DOWNLOADING
            )
        if current.status is DataFileStatus.DOWNLOADING:
            current = self._state_store.transition_data_file(
                current.file_id, DataFileStatus.DOWNLOADED, checksum=checksum
            )
        if current.status in {DataFileStatus.DOWNLOADED, DataFileStatus.VALIDATED}:
            self._state_store.transition_data_file(
                current.file_id,
                DataFileStatus.VALIDATED,
                checksum=checksum,
                row_count=row_count,
            )

    async def _download_small(
        self, session: aiohttp.ClientSession, url: str, *, max_bytes: int
    ) -> bytes:
        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with session.get(url, timeout=self._timeout) as response:
                    if response.status == 404:
                        raise ArchiveError(f"official archive not found: {url}")
                    if (
                        response.status == 429 or 500 <= response.status < 600
                    ) and attempt < self._max_attempts:
                        await self._sleep(self._retry_delay(attempt, response.headers))
                        continue
                    if response.status != 200:
                        raise ArchiveError(f"HTTP {response.status} downloading {url}")
                    payload = bytearray()
                    async for chunk in response.content.iter_chunked(min(max_bytes, 65_536)):
                        if len(payload) + len(chunk) > max_bytes:
                            raise ArchiveError(f"response exceeded {max_bytes} bytes: {url}")
                        payload.extend(chunk)
                    return bytes(payload)
            except (aiohttp.ClientError, TimeoutError) as exc:
                last_error = exc
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(attempt, {}))
                    continue
        raise ArchiveError(
            f"download failed after {self._max_attempts} attempts: {url}: {last_error}"
        ) from last_error

    async def _download_archive(
        self,
        session: aiohttp.ClientSession,
        url: str,
        *,
        archive_path: Path,
        expected_checksum: str,
    ) -> tuple[int, bool]:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = archive_path.with_name(f"{archive_path.name}.part")
        total_downloaded = 0
        ever_resumed = False
        last_error: BaseException | None = None

        if (
            part_path.exists()
            and sha256_file(part_path, chunk_bytes=self._chunk_bytes) == expected_checksum
        ):
            os.replace(part_path, archive_path)
            return 0, True

        for attempt in range(1, self._max_attempts + 1):
            offset = part_path.stat().st_size if part_path.exists() else 0
            if offset > self._max_archive_bytes:
                self._quarantine_path(part_path, "oversized")
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                async with session.get(url, headers=headers, timeout=self._timeout) as response:
                    if response.status == 404:
                        raise ArchiveError(f"official archive not found: {url}")
                    if response.status == 416 and offset:
                        self._quarantine_path(part_path, "range-not-satisfiable")
                        if attempt < self._max_attempts:
                            continue
                    if (
                        response.status == 429 or 500 <= response.status < 600
                    ) and attempt < self._max_attempts:
                        await self._sleep(self._retry_delay(attempt, response.headers))
                        continue
                    if response.status not in {200, 206}:
                        raise ArchiveError(f"HTTP {response.status} downloading {url}")

                    append = offset > 0 and response.status == 206
                    if append:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ArchiveError(
                                f"invalid Content-Range while resuming {url}: {content_range!r}"
                            )
                        ever_resumed = True
                    mode = "ab" if append else "wb"
                    current_size = offset if append else 0
                    with part_path.open(mode) as stream:
                        async for chunk in response.content.iter_chunked(self._chunk_bytes):
                            current_size += len(chunk)
                            total_downloaded += len(chunk)
                            if current_size > self._max_archive_bytes:
                                raise ArchiveError(
                                    f"archive exceeded {self._max_archive_bytes} bytes: {url}"
                                )
                            stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())

                actual_checksum = sha256_file(part_path, chunk_bytes=self._chunk_bytes)
                if actual_checksum != expected_checksum:
                    self._quarantine_path(part_path, f"checksum-{actual_checksum[:12]}")
                    if attempt < self._max_attempts:
                        continue
                    raise ArchiveError(
                        f"checksum mismatch for {url}: expected {expected_checksum}, "
                        f"received {actual_checksum}"
                    )
                os.replace(part_path, archive_path)
                return total_downloaded, ever_resumed
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(attempt, {}))
                    continue
        raise ArchiveError(
            f"archive download failed after {self._max_attempts} attempts: {url}: {last_error}"
        ) from last_error

    def _write_checksum(self, target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                if target.read_bytes() == payload:
                    return
            except OSError as exc:
                raise ArchiveError(f"cannot read checksum {target}: {exc}") from exc
            self._quarantine_path(target, "replaced-checksum")
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            raise ArchiveError(f"cannot persist checksum {target}: {exc}") from exc
        finally:
            if temporary.exists():
                temporary.unlink()

    def _quarantine_existing(
        self,
        target: ArchiveTarget,
        archive_path: Path,
        checksum_path: Path,
        extracted_path: Path,
    ) -> None:
        for path in (archive_path, checksum_path, extracted_path):
            if path.exists():
                self._quarantine_path(path, f"invalid-{target.day.isoformat()}")

    def _quarantine_path(self, path: Path, reason: str) -> Path:
        quarantine = self._storage.path(
            "quarantine",
            "archive_downloads",
            datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ"),
            f"{path.name}.{reason}",
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, quarantine)
        except OSError as exc:
            raise ArchiveError(f"cannot quarantine {path}: {exc}") from exc
        return quarantine

    def _archive_path(self, target: ArchiveTarget) -> Path:
        return self._storage.path(
            "raw_archives",
            "binance",
            "usdm",
            target.dataset,
            target.symbol,
            target.interval,
            target.filename,
        )

    @staticmethod
    def _file_id(target: ArchiveTarget) -> str:
        return deterministic_file_id(
            "binance_public_data",
            "usdm",
            "daily",
            target.dataset,
            target.symbol,
            target.interval,
            target.day.isoformat(),
        )

    def _retry_delay(self, attempt: int, headers: Mapping[str, str]) -> float:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        exponential: float = self._retry_base_seconds * (2 ** (attempt - 1))
        delay = exponential + float(self._jitter(self._retry_base_seconds))
        return delay if delay < 60.0 else 60.0
