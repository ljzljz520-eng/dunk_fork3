"""Explicit, verifiable providers for the two sides of a unified diff.

The renderer must never silently stitch together hunk text and bytes read
from the current working tree: those can be two completely different
snapshots (an old ``git show`` diff, a patch produced on another machine,
further edits made after the diff was generated, ...).

`PatchSideProvider` resolves an immutable `DualSnapshot` per patched file,
using, in order of preference:

1. Blob objects named by the ``index <old>..<new>`` line, fetched from the
   git object store with ``git cat-file --batch``. One side is enough: the
   other side is produced by *strictly* applying the patch hunks and the
   resulting blob id is recomputed (SHA-1) and compared with the id in the
   index line.
2. The working tree, but only as a content-hash-verified optimization:
   ``git hash-object`` runs the configured clean/filters and the raw bytes
   must hash to the blob id advertised by the patch. CRLF smudging is
   reversed explicitly; opaque smudge filters are never guessed.
3. A *partial* dual-side view assembled solely from the hunk/context lines
   of the patch. Unknown regions are blank placeholders that preserve line
   numbering. No working-tree bytes are involved.

Every failure that is specific to one file is captured on the returned
`DualSnapshot` instead of being raised, so one inconsistent file can never
poison the global statistics or the rendering of the other files.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from unidiff.constants import LINE_TYPE_NO_NEWLINE
from unidiff.patch import Hunk, PatchedFile

ZERO_OID = "0" * 40
FULL_OID_LEN = 40

_RE_INDEX = re.compile(
    r"^index "
    r"(?P<old>[0-9a-f]{4,40})(?:\.\.(?P<new>[0-9a-f]{4,40}))?"
    r"(?: (?P<mode>[0-7]{6}))?\s*$"
)
_RE_NEW_FILE_MODE = re.compile(r"^new file mode (?P<mode>[0-7]{6})\s*$")
_RE_DELETED_FILE_MODE = re.compile(r"^deleted file mode (?P<mode>[0-7]{6})\s*$")
_RE_OLD_MODE = re.compile(r"^old mode (?P<mode>[0-7]{6})\s*$")
_RE_NEW_MODE = re.compile(r"^new mode (?P<mode>[0-7]{6})\s*$")
_RE_RENAME_FROM = re.compile(r"^rename from (?P<path>.+)$")
_RE_RENAME_TO = re.compile(r"^rename to (?P<path>.+)$")


# ---------------------------------------------------------------------------
# Parsed git metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitIndexInfo:
    """Information carried by the git-specific patch header lines."""

    old_oid: Optional[str] = None
    new_oid: Optional[str] = None
    old_mode: Optional[str] = None
    new_mode: Optional[str] = None
    new_file_mode: Optional[str] = None
    deleted_file_mode: Optional[str] = None
    rename_from: Optional[str] = None
    rename_to: Optional[str] = None

    @staticmethod
    def parse(patch_info_lines: Iterable[str]) -> Optional["GitIndexInfo"]:
        values: Dict[str, str] = {}
        seen_git_header = False
        for raw in patch_info_lines:
            line = raw.rstrip("\n")
            if line.startswith("diff --git "):
                seen_git_header = True
                continue
            match = _RE_INDEX.match(line)
            if match:
                values["old_oid"] = match.group("old")
                if match.group("new") is not None:
                    values["new_oid"] = match.group("new")
                mode = match.group("mode")
                if mode is not None:
                    values["old_mode"] = mode
                    values["new_mode"] = mode
                continue
            for regex, key in (
                (_RE_NEW_FILE_MODE, "new_file_mode"),
                (_RE_DELETED_FILE_MODE, "deleted_file_mode"),
                (_RE_OLD_MODE, "old_mode"),
                (_RE_NEW_MODE, "new_mode"),
            ):
                match = regex.match(line)
                if match:
                    values[key] = match.group("mode")
                    continue
            match = _RE_RENAME_FROM.match(line)
            if match:
                values["rename_from"] = match.group("path").strip()
                continue
            match = _RE_RENAME_TO.match(line)
            if match:
                values["rename_to"] = match.group("path").strip()
                continue
        if not seen_git_header and "old_oid" not in values:
            return None
        return GitIndexInfo(**values)


def _is_abbreviated(oid: str) -> bool:
    return len(oid) < FULL_OID_LEN


def _is_zero(oid: Optional[str]) -> bool:
    return oid is None or oid == ZERO_OID or set(oid) == {"0"}


# ---------------------------------------------------------------------------
# Snapshot data model
# ---------------------------------------------------------------------------


class Provenance(str, Enum):
    """Where a file snapshot's bytes came from / how strongly verified."""

    GIT_BLOB = "git-blob"
    # Derived from the opposite side by applying hunks; recomputed blob id
    # matches the index line exactly.
    HUNK_DERIVED = "hunk-derived"
    # Read from the working tree; raw/cleaned bytes hash to the expected id.
    WORKTREE_HASHED = "worktree-hashed"
    # Reassembled from the patch's own hunk/context lines; not a full file.
    PATCH_CONTEXT = "patch-context"
    # The side provably does not exist (zero blob id / /dev/null).
    EMPTY = "empty"


class TagLevel(str, Enum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


class SnapshotTag(NamedTuple):
    level: TagLevel
    text: str


@dataclass(frozen=True)
class FileSnapshot:
    """One immutable side ("old"/source or "new"/target) of a patched file."""

    text: str
    provenance: Provenance
    complete: bool
    line_count: int
    oid: Optional[str] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def lines(self) -> List[str]:
        return split_lines_preserving_endings(self.text)


@dataclass(frozen=True)
class DualSnapshot:
    """Both sides of one patched file, or an isolated error for that file."""

    source: Optional[FileSnapshot]
    target: Optional[FileSnapshot]
    tags: Tuple[SnapshotTag, ...] = ()
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.source is not None and self.target is not None

    @classmethod
    def failure(cls, message: str) -> "DualSnapshot":
        return cls(
            source=None,
            target=None,
            tags=(SnapshotTag(TagLevel.ERROR, "snapshot error"),),
            error=message,
        )


# ---------------------------------------------------------------------------
# Line/blob level helpers
# ---------------------------------------------------------------------------


def split_lines_preserving_endings(text: str) -> List[str]:
    """Split text the way git treats text files: ``\\n`` is the separator.

    ``\\r\\n`` stays attached to its line, and a final line without a
    trailing newline is returned verbatim (no synthetic newline).
    """
    if text == "":
        return []
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1] != "":
        # Text does not end with a newline: the final piece is a real line.
        lines.append(parts[-1])
    return lines


def blob_id(data: bytes) -> str:
    """Compute git's blob object id for raw (clean-filtered) file bytes."""
    header = b"blob " + str(len(data)).encode("ascii") + b"\x00"
    return hashlib.sha1(header + data).hexdigest()


def encode_blob_text(text: str) -> bytes:
    """Encode reconstructed text back to blob bytes.

    ``surrogateescape`` lets bytes that survived an UTF-8 decode round-trip
    losslessly; git blob hashing is byte based.
    """
    return text.encode("utf-8", "surrogateescape")


class HunkApplicationError(ValueError):
    """A hunk could not be strictly applied at its declared position."""


def _hunk_side_lines(
    hunks: Sequence[Hunk], side: str
) -> Iterable[Tuple[Hunk, int, int, List[str]]]:
    for hunk in hunks:
        if side == "source":
            start, length = hunk.source_start, hunk.source_length
            values = _corrected_side_values(hunk, "source")
        else:
            start, length = hunk.target_start, hunk.target_length
            values = _corrected_side_values(hunk, "target")
        yield hunk, start, length, values


def _corrected_side_values(hunk: Hunk, side: str) -> List[str]:
    """Return this side's line values with no-newline-at-EOF honoured.

    ``unidiff`` keeps the diff stream's own trailing newline on every hunk
    body line, but a ``\\ No newline at end of file`` marker means the
    corresponding file line actually has no terminator. Strip it for the
    side(s) the marker applies to.
    """
    lines = list(hunk)
    values: List[str] = []
    for index, line in enumerate(lines):
        if line.line_type == LINE_TYPE_NO_NEWLINE:
            continue
        if side == "source":
            relevant = line.is_context or line.is_removed
        else:
            relevant = line.is_context or line.is_added
        if not relevant:
            continue
        value = line.value
        next_is_marker = (
            index + 1 < len(lines)
            and lines[index + 1].line_type == LINE_TYPE_NO_NEWLINE
        )
        if next_is_marker:
            if value.endswith("\r\n"):
                value = value[:-2]
            elif value.endswith("\n"):
                value = value[:-1]
        values.append(value)
    return values


def apply_hunks(base_text: str, hunks: Sequence[Hunk], *, forward: bool) -> str:
    """Strictly apply (or reverse) hunks to a complete file text.

    Hunks must match at the positions declared by their headers (no fuzz,
    no offset search). Multi-hunk shifts are handled by walking the file
    forwards and copying the gaps between successive hunks. A hunk with a
    zero side length (pure insertion / deletion) anchors *after* the line
    named in its header, matching the unified diff convention.
    """
    base_lines = split_lines_preserving_endings(base_text)
    out: List[str] = []
    position = 0
    for hunk in hunks:
        if forward:
            start, length = hunk.source_start, hunk.source_length
            removed = _corrected_side_values(hunk, "source")
            inserted = _corrected_side_values(hunk, "target")
        else:
            start, length = hunk.target_start, hunk.target_length
            removed = _corrected_side_values(hunk, "target")
            inserted = _corrected_side_values(hunk, "source")

        # 1-based header coordinate of the first hunk line; for a zero
        # length side the header names the line *before* which nothing is
        # taken, i.e. exactly `start` preceding lines.
        anchor = start if length == 0 else start - 1
        anchor = max(anchor, 0)

        if anchor < position:
            raise HunkApplicationError(
                f"hunk at line {start} overlaps a previously applied hunk"
            )
        if anchor + length > len(base_lines):
            raise HunkApplicationError(
                f"hunk at line {start} runs past end of file "
                f"(file has {len(base_lines)} lines)"
            )

        window = base_lines[anchor : anchor + length]
        if window != removed:
            mismatch = _first_mismatch(window, removed)
            raise HunkApplicationError(
                f"hunk context does not match file contents at line "
                f"{start + mismatch}: patch expects "
                f"{removed[mismatch]!r}, file has {window[mismatch]!r}"
            )

        out.extend(base_lines[position:anchor])
        out.extend(inserted)
        position = anchor + length

    out.extend(base_lines[position:])
    return "".join(out)


def _first_mismatch(actual: Sequence[str], expected: Sequence[str]) -> int:
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            return index
    return min(len(actual), len(expected))


def partial_side_from_hunks(hunks: Sequence[Hunk], side: str) -> Tuple[str, int]:
    """Build a line-number-aligned partial view from patch text only.

    Unknown regions are filled with empty placeholder lines so that
    ``Syntax(line_range=...)`` and the gutter show the same line numbers as
    the real file. The result is marked incomplete by the caller and is
    never extended with working-tree bytes.
    """
    out: List[str] = []
    cursor = 0
    for _, start, length, values in _hunk_side_lines(hunks, side):
        anchor = max(start if length == 0 else start - 1, 0)
        if anchor > cursor:
            out.extend(["\n"] * (anchor - cursor))
        out.extend(values)
        cursor = anchor + len(values)
    return "".join(out), cursor


# ---------------------------------------------------------------------------
# Git plumbing access (blobs + hash verification only; no `git show`)
# ---------------------------------------------------------------------------


class GitObjectAccess:
    """Minimal, batch-oriented access to git blob objects.

    Only ``rev-parse``, ``cat-file --batch`` and ``hash-object`` are used:
    this is deliberately not a general ``git show <rev>:<path>`` frontend.
    Blob ids always come from the patch being rendered.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._proc: Optional[subprocess.Popen] = None

    @classmethod
    def open(cls) -> Optional["GitObjectAccess"]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        root = Path(result.stdout.decode("utf-8", "replace").strip())
        if not root.is_dir():
            return None
        return cls(root)

    def _ensure_batch(self) -> Optional[subprocess.Popen]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        try:
            self._proc = subprocess.Popen(
                ["git", "-C", str(self.repo_root), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            self._proc = None
        return self._proc

    def get_blobs(self, oids: Iterable[str]) -> Dict[str, bytes]:
        """Fetch blobs keyed by both requested and fully resolved oid."""
        unique_oids = tuple(dict.fromkeys(oids))
        if not unique_oids:
            return {}
        proc = self._ensure_batch()
        if proc is None or proc.stdin is None or proc.stdout is None:
            return {}

        blobs: Dict[str, bytes] = {}
        for requested in unique_oids:
            try:
                proc.stdin.write(requested.encode("ascii") + b"\n")
                proc.stdin.flush()
                header = proc.stdout.readline()
            except (BrokenPipeError, OSError):
                break
            if not header:
                break
            parts = header.decode("utf-8", "replace").rstrip("\n").split(" ")
            # "<oid> blob <size>", "<oid> missing" or "<oid> <errtype>"
            if len(parts) < 3 or parts[1] != "blob":
                continue
            resolved_oid, size_text = parts[0], parts[2]
            try:
                size = int(size_text)
            except ValueError:
                continue
            data = self._read_exact(proc.stdout, size)
            if data is None:
                break
            # Every response is followed by a single newline delimiter.
            proc.stdout.read(1)
            blobs[requested] = data
            blobs[resolved_oid] = data
        return blobs

    @staticmethod
    def _read_exact(stream, size: int) -> Optional[bytes]:
        chunks: List[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def hash_object(self, path: Path) -> Optional[str]:
        """Hash worktree bytes *through the configured clean filters*.

        Equivalent to the id git would store after ``git add``; nothing is
        written to the object database (no ``-w``).
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_root), "hash-object", "--", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        oid = result.stdout.decode("ascii", "replace").strip()
        return oid if re.fullmatch(r"[0-9a-f]{40}", oid) else None

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
            self._proc = None

    def __enter__(self) -> "GitObjectAccess":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# The provider itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VerifiedText:
    text: str
    oid: str
    note: Optional[str] = None


class PatchSideProvider:
    """Resolve immutable, provenance-tracked `DualSnapshot`s for patches."""

    def __init__(
        self,
        git: Optional[GitObjectAccess] = None,
        root: Optional[Path] = None,
    ):
        self.git = git
        self.root = root if root is not None else (
            git.repo_root if git is not None else None
        )

    @classmethod
    def open(cls) -> "PatchSideProvider":
        git = GitObjectAccess.open()
        return cls(git=git)

    def close(self) -> None:
        if self.git is not None:
            self.git.close()

    def __enter__(self) -> "PatchSideProvider":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- public API --------------------------------------------------------

    def snapshot(self, patch: PatchedFile) -> DualSnapshot:
        """Return the two sides for *patch*; never raises for file issues."""
        try:
            return self._snapshot(patch)
        except Exception as exc:  # isolated per file; never fatal globally
            return DualSnapshot.failure(f"{type(exc).__name__}: {exc}")

    # -- internals ---------------------------------------------------------

    def _snapshot(self, patch: PatchedFile) -> DualSnapshot:
        hunks = list(patch)
        info = GitIndexInfo.parse(patch.patch_info or [])

        old_oid = info.old_oid if info else None
        new_oid = info.new_oid if info else None
        if patch.is_added_file:
            old_oid = ZERO_OID
        if patch.is_removed_file:
            new_oid = ZERO_OID

        old_path, new_path = self._side_paths(patch, info, self.root)

        wanted = {
            oid
            for oid in (old_oid, new_oid)
            if oid and not _is_zero(oid)
        }
        blobs = self.git.get_blobs(wanted) if (self.git and wanted) else {}

        old_bytes = self._lookup_blob(blobs, old_oid)
        new_bytes = self._lookup_blob(blobs, new_oid)

        if old_bytes is not None:
            return self._snapshot_from_source_blob(
                old_bytes, old_oid, new_bytes, new_oid, hunks
            )
        if new_bytes is not None:
            return self._snapshot_from_target_blob(
                old_oid, new_bytes, new_oid, hunks
            )
        return self._snapshot_without_blobs(
            old_oid, new_oid, old_path, new_path, hunks
        )

    @staticmethod
    def _lookup_blob(
        blobs: Dict[str, bytes], oid: Optional[str]
    ) -> Optional[bytes]:
        if not oid or _is_zero(oid):
            return None
        return blobs.get(oid)

    @staticmethod
    def _side_paths(
        patch: PatchedFile, info: Optional[GitIndexInfo], root: Optional[Path]
    ) -> Tuple[Optional[Path], Optional[Path]]:
        def resolve(label_path: str, renamed: Optional[str]) -> Optional[Path]:
            if label_path in (None, "/dev/null"):
                return None
            relative = renamed if renamed is not None else _strip_prefix(label_path)
            if relative is None or root is None:
                return None
            return root / relative

        return (
            resolve(patch.source_file, info.rename_from if info else None),
            resolve(patch.target_file, info.rename_to if info else None),
        )

    # -- blob anchored resolutions ----------------------------------------

    def _snapshot_from_source_blob(
        self,
        old_bytes: bytes,
        old_oid: Optional[str],
        new_bytes: Optional[bytes],
        new_oid: Optional[str],
        hunks: Sequence[Hunk],
    ) -> DualSnapshot:
        notes: List[str] = []
        old_text = old_bytes.decode("utf-8", "surrogateescape")

        if not _is_zero(old_oid) and not self._oid_matches(blob_id(old_bytes), old_oid):
            return DualSnapshot.failure(
                f"source blob content hash does not match index id {old_oid}"
            )

        source = FileSnapshot(
            text=old_text,
            provenance=Provenance.GIT_BLOB,
            complete=True,
            line_count=len(split_lines_preserving_endings(old_text)),
            oid=old_oid,
        )

        if _is_zero(new_oid):
            # Deletion: no target side.
            target = FileSnapshot(
                text="",
                provenance=Provenance.EMPTY,
                complete=True,
                line_count=0,
                oid=ZERO_OID,
            )
            return DualSnapshot(source=source, target=target, tags=tuple())

        try:
            derived_text = apply_hunks(old_text, hunks, forward=True)
        except HunkApplicationError as exc:
            return DualSnapshot.failure(f"hunks do not apply to source blob: {exc}")

        derived_bytes = encode_blob_text(derived_text)
        derived_id = blob_id(derived_bytes)
        if not self._oid_matches(derived_id, new_oid):
            return DualSnapshot.failure(
                f"recomputed target blob {derived_id} does not match index "
                f"id {new_oid}"
            )
        if _is_abbreviated(new_oid):
            notes.append(
                f"target id {new_oid} is abbreviated; prefix-verified only"
            )

        if new_bytes is not None:
            # Independently cross-checked against a second stored object.
            if not self._oid_matches(blob_id(new_bytes), new_oid):
                return DualSnapshot.failure(
                    "target blob content hash does not match index id"
                )
            new_text = new_bytes.decode("utf-8", "surrogateescape")
            if new_text != derived_text:
                return DualSnapshot.failure(
                    "target blob and hunk-derived target disagree"
                )
            target_provenance = Provenance.GIT_BLOB
        else:
            new_text = derived_text
            target_provenance = Provenance.HUNK_DERIVED

        target = FileSnapshot(
            text=new_text,
            provenance=target_provenance,
            complete=True,
            line_count=len(split_lines_preserving_endings(new_text)),
            oid=(new_oid if new_oid and not _is_abbreviated(new_oid) else derived_id),
            notes=tuple(notes),
        )
        return DualSnapshot(
            source=source,
            target=target,
            tags=self._verified_tags(source, target),
        )

    def _snapshot_from_target_blob(
        self,
        old_oid: Optional[str],
        new_bytes: bytes,
        new_oid: Optional[str],
        hunks: Sequence[Hunk],
    ) -> DualSnapshot:
        new_text = new_bytes.decode("utf-8", "surrogateescape")
        if not _is_zero(new_oid) and not self._oid_matches(blob_id(new_bytes), new_oid):
            return DualSnapshot.failure(
                f"target blob content hash does not match index id {new_oid}"
            )

        target = FileSnapshot(
            text=new_text,
            provenance=Provenance.GIT_BLOB,
            complete=True,
            line_count=len(split_lines_preserving_endings(new_text)),
            oid=new_oid,
        )

        if _is_zero(old_oid):
            # Addition: source side is empty; prove the patch reproduces the
            # stored blob before accepting it.
            try:
                derived = apply_hunks("", hunks, forward=True)
            except HunkApplicationError as exc:
                return DualSnapshot.failure(
                    f"hunks do not reproduce added blob from empty file: {exc}"
                )
            if derived != new_text:
                return DualSnapshot.failure(
                    "added blob does not match content described by hunks"
                )
            source = FileSnapshot(
                text="",
                provenance=Provenance.EMPTY,
                complete=True,
                line_count=0,
                oid=ZERO_OID,
            )
            return DualSnapshot(
                source=source,
                target=target,
                tags=self._verified_tags(source, target),
            )

        try:
            derived_old = apply_hunks(new_text, hunks, forward=False)
        except HunkApplicationError as exc:
            return DualSnapshot.failure(
                f"hunks do not reverse-apply to target blob: {exc}"
            )
        derived_id = blob_id(encode_blob_text(derived_old))
        if not self._oid_matches(derived_id, old_oid):
            return DualSnapshot.failure(
                f"recomputed source blob {derived_id} does not match index "
                f"id {old_oid}"
            )
        notes: Tuple[str, ...] = ()
        if _is_abbreviated(old_oid):
            notes = (f"source id {old_oid} is abbreviated; prefix-verified only",)
        source = FileSnapshot(
            text=derived_old,
            provenance=Provenance.HUNK_DERIVED,
            complete=True,
            line_count=len(split_lines_preserving_endings(derived_old)),
            oid=(old_oid if old_oid and not _is_abbreviated(old_oid) else derived_id),
            notes=notes,
        )
        return DualSnapshot(
            source=source,
            target=target,
            tags=self._verified_tags(source, target),
        )

    # -- worktree / patch-context fallbacks --------------------------------

    def _snapshot_without_blobs(
        self,
        old_oid: Optional[str],
        new_oid: Optional[str],
        old_path: Optional[Path],
        new_path: Optional[Path],
        hunks: Sequence[Hunk],
    ) -> DualSnapshot:
        tags: List[SnapshotTag] = []
        notes: List[str] = []

        verified_target: Optional[_VerifiedText] = None
        if not _is_zero(new_oid) and new_path is not None:
            verified_target = self._verified_worktree_text(new_path, new_oid)
            if verified_target is None:
                tags.append(
                    SnapshotTag(
                        TagLevel.ERROR,
                        "worktree drifted: target bytes fail hash verification",
                    )
                )

        verified_source: Optional[_VerifiedText] = None
        if (
            verified_target is None
            and not _is_zero(old_oid)
            and old_path is not None
            and old_path != new_path
        ):
            verified_source = self._verified_worktree_text(old_path, old_oid)
            if verified_source is None and not tags:
                tags.append(
                    SnapshotTag(
                        TagLevel.ERROR,
                        "worktree drifted: source bytes fail hash verification",
                    )
                )

        if verified_target is not None:
            target = FileSnapshot(
                text=verified_target.text,
                provenance=Provenance.WORKTREE_HASHED,
                complete=True,
                line_count=len(
                    split_lines_preserving_endings(verified_target.text)
                ),
                oid=verified_target.oid,
                notes=((verified_target.note,) if verified_target.note else ()),
            )
            if _is_zero(old_oid):
                try:
                    if apply_hunks("", hunks, forward=True) != verified_target.text:
                        return DualSnapshot.failure(
                            "hash-verified worktree target disagrees with hunks"
                        )
                except HunkApplicationError as exc:
                    return DualSnapshot.failure(f"hunks do not match worktree: {exc}")
                source = FileSnapshot(
                    "", Provenance.EMPTY, True, 0, ZERO_OID
                )
            else:
                try:
                    old_text = apply_hunks(
                        verified_target.text, hunks, forward=False
                    )
                except HunkApplicationError as exc:
                    return DualSnapshot.failure(
                        f"hunks do not reverse-apply to verified worktree: {exc}"
                    )
                derived_id = blob_id(encode_blob_text(old_text))
                if not self._oid_matches(derived_id, old_oid):
                    return DualSnapshot.failure(
                        f"recomputed source blob {derived_id} does not match "
                        f"index id {old_oid}"
                    )
                source = FileSnapshot(
                    text=old_text,
                    provenance=Provenance.HUNK_DERIVED,
                    complete=True,
                    line_count=len(split_lines_preserving_endings(old_text)),
                    oid=(
                        old_oid
                        if old_oid and not _is_abbreviated(old_oid)
                        else derived_id
                    ),
                )
            if verified_target.note:
                notes.append(verified_target.note)
            tags = [
                SnapshotTag(TagLevel.OK, "snapshot: worktree (content-hash verified)")
            ]
            return DualSnapshot(source=source, target=target, tags=tuple(tags))

        if verified_source is not None:
            source = FileSnapshot(
                text=verified_source.text,
                provenance=Provenance.WORKTREE_HASHED,
                complete=True,
                line_count=len(
                    split_lines_preserving_endings(verified_source.text)
                ),
                oid=verified_source.oid,
            )
            if _is_zero(new_oid):
                target = FileSnapshot("", Provenance.EMPTY, True, 0, ZERO_OID)
            else:
                try:
                    new_text = apply_hunks(
                        verified_source.text, hunks, forward=True
                    )
                except HunkApplicationError as exc:
                    return DualSnapshot.failure(
                        f"hunks do not apply to verified worktree: {exc}"
                    )
                derived_id = blob_id(encode_blob_text(new_text))
                if not self._oid_matches(derived_id, new_oid):
                    return DualSnapshot.failure(
                        f"recomputed target blob {derived_id} does not match "
                        f"index id {new_oid}"
                    )
                target = FileSnapshot(
                    text=new_text,
                    provenance=Provenance.HUNK_DERIVED,
                    complete=True,
                    line_count=len(split_lines_preserving_endings(new_text)),
                    oid=(
                        new_oid
                        if new_oid and not _is_abbreviated(new_oid)
                        else derived_id
                    ),
                )
            tags = [
                SnapshotTag(TagLevel.OK, "snapshot: worktree (content-hash verified)")
            ]
            return DualSnapshot(source=source, target=target, tags=tuple(tags))

        return self._patch_context_snapshot(old_oid, new_oid, hunks, tags)

    def _verified_worktree_text(
        self, path: Path, expected_oid: Optional[str]
    ) -> Optional[_VerifiedText]:
        """Return worktree content only if its bytes hash to *expected_oid*.

        Honors clean filters through ``git hash-object`` and explicitly
        reverses a CRLF smudge; opaque filters are rejected rather than
        guessed.
        """
        if self.git is None or expected_oid is None or _is_zero(expected_oid):
            return None
        if not path.is_file():
            return None
        try:
            raw = path.read_bytes()
        except OSError:
            return None

        raw_oid = blob_id(raw)
        if self._oid_matches(raw_oid, expected_oid):
            return _VerifiedText(
                raw.decode("utf-8", "surrogateescape"), oid=raw_oid
            )

        clean_oid = self.git.hash_object(path)
        normalized = raw.replace(b"\r\n", b"\n")
        normalized_id = blob_id(normalized)
        if (
            clean_oid is not None
            and self._oid_matches(clean_oid, expected_oid)
            and self._oid_matches(normalized_id, expected_oid)
        ):
            return _VerifiedText(
                normalized.decode("utf-8", "surrogateescape"),
                oid=normalized_id,
                note="CRLF worktree bytes normalized to blob line endings",
            )
        # Any other outcome (drift, or an opaque smudge filter we cannot
        # invert) must not let worktree content into the snapshot.
        return None

    def _patch_context_snapshot(
        self,
        old_oid: Optional[str],
        new_oid: Optional[str],
        hunks: Sequence[Hunk],
        tags: List[SnapshotTag],
    ) -> DualSnapshot:
        source_text, source_lines = partial_side_from_hunks(hunks, "source")
        target_text, target_lines = partial_side_from_hunks(hunks, "target")
        source = FileSnapshot(
            text=source_text,
            provenance=Provenance.PATCH_CONTEXT,
            complete=False,
            line_count=source_lines,
            oid=(None if _is_zero(old_oid) else old_oid),
        )
        target = FileSnapshot(
            text=target_text,
            provenance=Provenance.PATCH_CONTEXT,
            complete=False,
            line_count=target_lines,
            oid=(None if _is_zero(new_oid) else new_oid),
        )
        if not tags:
            tags = [
                SnapshotTag(
                    TagLevel.WARN,
                    "partial snapshot: patch context only, worktree not consulted",
                )
            ]
        else:
            tags.append(
                SnapshotTag(
                    TagLevel.WARN,
                    "partial snapshot: patch context only",
                )
            )
        return DualSnapshot(source=source, target=target, tags=tuple(tags))

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _oid_matches(actual_full_oid: str, expected: Optional[str]) -> bool:
        if expected is None:
            # No id advertised: the hunk round-trip itself is the only check.
            return True
        if _is_zero(expected):
            return False
        if _is_abbreviated(expected):
            return actual_full_oid.startswith(expected)
        return actual_full_oid == expected

    @staticmethod
    def _verified_tags(
        source: FileSnapshot, target: FileSnapshot
    ) -> Tuple[SnapshotTag, ...]:
        provenances = {source.provenance, target.provenance}
        notes = sorted({note for side in (source, target) for note in side.notes})
        tags: List[SnapshotTag] = []
        if Provenance.GIT_BLOB in provenances:
            if provenances <= {Provenance.GIT_BLOB, Provenance.EMPTY}:
                label = "snapshot: git blobs verified"
            else:
                label = "snapshot: git blob + hunk-derived, hash verified"
            level = TagLevel.WARN if notes else TagLevel.OK
        else:
            label = "snapshot: hunk-derived, hash verified"
            level = TagLevel.WARN if notes else TagLevel.OK
        tags.append(SnapshotTag(level, label))
        for note in notes:
            tags.append(SnapshotTag(TagLevel.WARN, note))
        return tuple(tags)


def _strip_prefix(label_path: str) -> Optional[str]:
    """Strip the ``a/`` / ``b/`` diff label prefix."""
    if label_path in ("", "/dev/null"):
        return None
    if label_path.startswith(("a/", "b/")):
        return label_path[2:]
    return label_path
