"""Tests for dunk.patch_sides.PatchSideProvider.

Most scenario tests stand up an ephemeral git repository with a known
content history, capture a real `git diff` / `git diff --cached` string,
parse it with unidiff, and assert that the provider resolves two
verifiable, internally-consistent sides without ever falling back to the
untested working tree.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from unidiff import PatchSet

from dunk.patch_sides import (
    DualSnapshot,
    FileSnapshot,
    GitIndexInfo,
    GitObjectAccess,
    HunkApplicationError,
    Provenance,
    SnapshotTag,
    TagLevel,
    PatchSideProvider,
    ZERO_OID,
    apply_hunks,
    blob_id,
    partial_side_from_hunks,
    split_lines_preserving_endings,
)


# ---------------------------------------------------------------------------
# Tiny git harness
# ---------------------------------------------------------------------------


class GitRepo:
    def __init__(self, root: Path):
        self.root = root
        self.env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "dunk-test",
            "GIT_AUTHOR_EMAIL": "dunk@test",
            "GIT_COMMITTER_NAME": "dunk-test",
            "GIT_COMMITTER_EMAIL": "dunk@test",
        }

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            env=self.env,
        )
        return result.stdout.decode("utf-8", "surrogateescape")

    def write(self, name: str, content: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def blob(self, name: str) -> bytes:
        # The committed/staged blob of a path, for independent checks.
        return self.git("cat-file", "-p", f"HEAD:{name}").encode("utf-8", "surrogateescape")


@pytest.fixture()
def repo(tmp_path: Path) -> GitRepo:
    r = GitRepo(tmp_path / "repo")
    r.root.mkdir()
    r.git("init", "-q")
    r.git("config", "core.autocrlf", "false")
    r.git("config", "user.name", "dunk-test")
    r.git("config", "user.email", "dunk@test")
    return r


def parse(diff_text: str) -> PatchSet:
    return PatchSet(diff_text)


def first(patch_set: PatchSet):
    return patch_set[0]


def take(repo: GitRepo, diff: str):
    ps = parse(diff)
    assert len(ps) >= 1
    return first(ps)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_split_lines_preserving_endings():
    assert split_lines_preserving_endings("") == []
    assert split_lines_preserving_endings("a\nb\n") == ["a\n", "b\n"]
    assert split_lines_preserving_endings("a\nb") == ["a\n", "b"]
    assert split_lines_preserving_endings("\n") == ["\n"]
    assert split_lines_preserving_endings("a\r\nb\r\n") == ["a\r\n", "b\r\n"]


def test_apply_hunks_forward():
    original = "one\ntwo\nthree\n"
    hunks = list(
        parse(
            "--- a\n+++ b\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
        )[0]
    )
    assert apply_hunks(original, hunks, forward=True) == "one\nTWO\nthree\n"


def test_apply_hunks_reverse():
    original = "one\nTWO\nthree\n"
    hunks = list(
        parse(
            "--- a\n+++ b\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
        )[0]
    )
    assert apply_hunks(original, hunks, forward=False) == "one\ntwo\nthree\n"


def test_apply_hunks_strict_rejects_context_mismatch():
    original = "one\nWRONG\nthree\n"
    hunks = list(
        parse(
            "--- a\n+++ b\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
        )[0]
    )
    with pytest.raises(HunkApplicationError):
        apply_hunks(original, hunks, forward=True)


def test_partial_side_line_alignment():
    ps = parse(
        "--- a\n+++ b\n"
        "@@ -5,3 +5,3 @@\n"
        " ctx\n-old\n+new\n ctx\n"
        "@@ -20,2 +20,2 @@\n"
        " x\n-y\n+z\n"
    )
    hunks = list(ps[0])
    text, count = partial_side_from_hunks(hunks, "source")
    lines = split_lines_preserving_endings(text)
    # First hunk starts at source line 5 => 4 placeholder lines before it.
    assert len(lines) >= 21
    assert lines[4] == "ctx\n"
    assert lines[5] == "old\n"


# ---------------------------------------------------------------------------
# Index line parsing
# ---------------------------------------------------------------------------


def test_parse_index_info_full_oid():
    info = GitIndexInfo.parse(
        [
            "diff --git a/file b/file\n",
            "index 1111111111111111111111111111111111111111..2222222222222222222222222222222222222222 100644\n",
            "--- a/file\n",
            "+++ b/file\n",
        ]
    )
    assert info is not None
    assert info.old_oid == "1" * 40
    assert info.new_oid == "2" * 40
    assert info.old_mode == info.new_mode == "100644"


def test_parse_index_info_abbreviated_and_rename():
    info = GitIndexInfo.parse(
        [
            "diff --git a/old b/new\n",
            "similarity index 85%\n",
            "rename from old\n",
            "rename to new\n",
            "index abc1234..def5678\n",
            "--- a/old\n",
            "+++ b/new\n",
        ]
    )
    assert info.old_oid == "abc1234"
    assert info.new_oid == "def5678"
    assert info.rename_from == "old"
    assert info.rename_to == "new"


def test_parse_index_info_no_git_header_returns_none():
    assert GitIndexInfo.parse(["--- a\n", "+++ b\n"]) is None


# ---------------------------------------------------------------------------
# Real git scenarios
# ---------------------------------------------------------------------------


def _commit_baseline(repo: GitRepo):
    repo.write("file.txt", "alpha\nbeta\ngamma\n")
    repo.git("add", ".")
    repo.git("commit", "-q", "-m", "init")


def test_unstaged_modification_uses_blob_and_hunk_derived(repo: GitRepo):
    _commit_baseline(repo)
    repo.write("file.txt", "alpha\nBETA\ngamma\n")

    patch = take(repo, repo.git("diff"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)

    assert dual.ok
    assert dual.source.provenance == Provenance.GIT_BLOB
    assert dual.target.provenance == Provenance.HUNK_DERIVED
    assert dual.source.complete and dual.target.complete
    # Recomputed target id must equal the index line id (abbreviated by default).
    assert dual.target.oid.startswith(GitIndexInfo.parse(patch.patch_info).new_oid)
    assert dual.target.text == "alpha\nBETA\ngamma\n"
    assert dual.source.text == "alpha\nbeta\ngamma\n"
    # No worktree consulted: no error tag.
    assert not any(t.level == TagLevel.ERROR for t in dual.tags)


def test_staged_modification_cross_checks_both_blobs(repo: GitRepo):
    _commit_baseline(repo)
    repo.write("file.txt", "alpha\nBETA\ngamma\n")
    repo.git("add", "file.txt")

    patch = take(repo, repo.git("diff", "--cached"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)

    assert dual.ok
    assert dual.source.provenance == Provenance.GIT_BLOB
    assert dual.target.provenance == Provenance.GIT_BLOB  # both in object db
    assert dual.target.text == "alpha\nBETA\ngamma\n"
    assert dual.source.text == "alpha\nbeta\ngamma\n"


def test_new_unstaged_file_uses_hash_verified_worktree(repo: GitRepo):
    repo.write("added.txt", "hello\nworld\n")
    # `git diff` ignores untracked files; use intent-to-add so the patch
    # carries an index line with a zero old oid and a real new oid.
    repo.git("add", "-N", "added.txt")
    patch = take(repo, repo.git("diff"))

    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)

    assert dual.ok
    assert dual.source.provenance == Provenance.EMPTY
    assert dual.target.provenance == Provenance.WORKTREE_HASHED
    assert dual.target.text == "hello\nworld\n"
    assert blob_id(dual.target.text.encode("utf-8")) == dual.target.oid


def test_new_staged_file_uses_stored_blob(repo: GitRepo):
    repo.write("added.txt", "hello\nworld\n")
    repo.git("add", "added.txt")
    patch = take(repo, repo.git("diff", "--cached"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)
    assert dual.ok
    assert dual.source.provenance == Provenance.EMPTY
    assert dual.target.provenance == Provenance.GIT_BLOB
    assert dual.target.text == "hello\nworld\n"


def test_deletion_snapshot_uses_old_blob_and_empty_target(repo: GitRepo):
    _commit_baseline(repo)
    (repo.root / "file.txt").unlink()
    patch = take(repo, repo.git("diff"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)
    assert dual.ok
    assert dual.source.provenance == Provenance.GIT_BLOB
    assert dual.target.provenance == Provenance.EMPTY
    assert dual.source.text == "alpha\nbeta\ngamma\n"


def test_rename_with_change_resolves_from_old_blob(repo: GitRepo):
    _commit_baseline(repo)
    repo.git("mv", "file.txt", "renamed.txt")
    repo.write("renamed.txt", "alpha\nBETA\ngamma\n")
    repo.git("add", "-A")
    # Staged rename with content modification: old blob is in the db.
    patch = take(repo, repo.git("diff", "--cached", "-M"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)
    assert dual.ok
    assert dual.source.provenance == Provenance.GIT_BLOB
    assert dual.target.provenance in (
        Provenance.GIT_BLOB,
        Provenance.HUNK_DERIVED,
    )
    assert dual.source.text == "alpha\nbeta\ngamma\n"
    assert dual.target.text == "alpha\nBETA\ngamma\n"


def test_abbreviated_oids_prefix_verified(repo: GitRepo):
    _commit_baseline(repo)
    repo.write("file.txt", "alpha\nBETA\ngamma\n")
    patch = take(repo, repo.git("diff", "--abbrev=7"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)
    assert dual.ok
    # The stored oid is the resolved full id; the short id is in the index.
    short = GitIndexInfo.parse(patch.patch_info).new_oid
    assert dual.target.oid.startswith(short)
    assert any(
        "abbreviated" in t.text for t in dual.tags
    ) or any("abbreviated" in note for note in dual.target.notes)


def test_no_newline_at_eof_preserved(repo: GitRepo):
    repo.write("f.txt", "one\ntwo")  # no trailing newline
    repo.git("add", ".")
    repo.git("commit", "-q", "-m", "init")
    repo.write("f.txt", "one\nTWO")  # still no trailing newline
    patch = take(repo, repo.git("diff"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)
    assert dual.ok
    assert not dual.source.text.endswith("\n")
    assert not dual.target.text.endswith("\n")
    assert dual.target.text == "one\nTWO"


def test_crlf_worktree_normalized_with_filter(repo: GitRepo):
    # Configure the file to go through CRLF<->LF conversion.
    (repo.root / ".gitattributes").write_text("*.txt text eol=crlf\n")
    repo.git("add", ".gitattributes")
    repo.write("f.txt", "a\r\nb\r\n")
    repo.git("add", ".")
    repo.git("commit", "-q", "-m", "init")
    # Modify worktree (still CRLF), produce diff.
    repo.write("f.txt", "a\r\nB\r\n")
    patch = take(repo, repo.git("diff"))
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)
    assert dual.ok
    # Snapshot content lives in the clean (LF) space.
    assert dual.target.text == "a\nB\n"
    # The derived target must match the index oid.
    assert dual.target.oid.startswith(GitIndexInfo.parse(patch.patch_info).new_oid)


def test_worktree_drift_on_added_file_yields_error_not_worktree_bytes(repo: GitRepo):
    repo.write("added.txt", "hello\nworld\n")
    repo.git("add", "-N", "added.txt")
    diff_text = repo.git("diff")
    # Now drift: the patch is stale.
    repo.write("added.txt", "hello\nworld\ndrift\n")

    patch = take(repo, diff_text)
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)

    # Either we get a hash-verification error or a partial-context fallback;
    # crucially the target side must NOT contain the drifted bytes.
    assert dual.target is not None
    assert "drift" not in dual.target.text


def test_tampered_index_oid_is_isolated_error(repo: GitRepo):
    _commit_baseline(repo)
    repo.write("file.txt", "alpha\nBETA\ngamma\n")
    diff_text = repo.git("diff")
    # Corrupt the target oid advertised by git (match abbrev or full ids).
    bad = "f" * 40
    diff_text = re.sub(
        r"index [0-9a-f]+\.\.[0-9a-f]+",
        lambda m: f"index {m.group(0).split('..')[0].split(' ')[-1]}..{bad}",
        diff_text,
    )
    patch = take(repo, diff_text)
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    dual = provider.snapshot(patch)
    # No exception escapes; the single file is reported as failed.
    assert not dual.ok
    assert dual.error is not None
    assert "does not match" in dual.error


def test_error_isolation_between_files(repo: GitRepo):
    _commit_baseline(repo)
    repo.write("file.txt", "alpha\nBETA\ngamma\n")
    repo.write("other.txt", "x\ny\n")
    diff_text = repo.git("diff")

    # Corrupt only the first file's advertised target oid.
    bad = "f" * 40
    seen = {"count": 0}

    def repl(match):
        if seen["count"] == 0:
            seen["count"] += 1
            old = match.group(0).split("..")[0].split(" ")[-1]
            return f"index {old}..{bad}"
        return match.group(0)

    diff_text = re.sub(r"index [0-9a-f]+\.\.[0-9a-f]+", repl, diff_text)

    ps = parse(diff_text)
    provider = PatchSideProvider(git=GitObjectAccess(repo.root), root=repo.root)
    results = [provider.snapshot(p) for p in ps]

    # Exactly one file is reported as failed; the rest still resolve.
    failed = [r for r in results if not r.ok]
    ok = [r for r in results if r.ok]
    assert len(failed) == 1
    assert len(ok) == len(ps) - 1
    # Global patch statistics (computed from unidiff metadata) are
    # unaffected by the per-file consistency failure.
    assert ps.added and ps.removed


def test_non_git_patch_falls_back_to_patch_context():
    diff = (
        "--- foo.txt\n"
        "+++ foo.txt\n"
        "@@ -2,2 +2,2 @@\n"
        " a\n"
        "-b\n"
        "+B\n"
        " c\n"
    )
    patch = first(parse(diff))
    provider = PatchSideProvider(git=None, root=None)
    dual = provider.snapshot(patch)
    assert dual.ok
    assert dual.source.provenance == Provenance.PATCH_CONTEXT
    assert dual.target.provenance == Provenance.PATCH_CONTEXT
    assert not dual.source.complete
    # Line alignment: source hunk starts at 2 -> one placeholder + 2 hunk lines.
    assert dual.source.line_count >= 3
    src_lines = split_lines_preserving_endings(dual.source.text)
    assert src_lines[1] == "a\n"
    assert src_lines[2] == "b\n"
