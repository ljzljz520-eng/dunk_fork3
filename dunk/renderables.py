from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from rich.align import Align
from rich.console import Console
from rich.console import ConsoleOptions, RenderResult
from rich.markup import escape
from rich.rule import Rule
from rich.segment import Segment
from rich.table import Table
from rich.text import Text
from unidiff import PatchedFile

from dunk.patch_sides import SnapshotTag, TagLevel
from dunk.underline_bar import UnderlineBar


def simple_pluralise(word: str, number: int) -> str:
    if number == 1:
        return word
    else:
        return word + "s"


@dataclass
class PatchSetHeader:
    file_modifications: int
    file_additions: int
    file_removals: int
    line_additions: int
    line_removals: int

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        if self.file_modifications:
            yield Align.center(
                f"[blue]{self.file_modifications} {simple_pluralise('file', self.file_modifications)} changed"
            )
        if self.file_additions:
            yield Align.center(
                f"[green]{self.file_additions} {simple_pluralise('file', self.file_additions)} added"
            )
        if self.file_removals:
            yield Align.center(
                f"[red]{self.file_removals} {simple_pluralise('file', self.file_removals)} removed"
            )

        bar_width = console.width // 5
        changed_lines = max(1, self.line_additions + self.line_removals)
        added_lines_ratio = self.line_additions / changed_lines

        line_changes_summary = Table.grid()
        line_changes_summary.add_column()
        line_changes_summary.add_column()
        line_changes_summary.add_column()
        line_changes_summary.add_row(
            f"[bold green]+{self.line_additions} ",
            UnderlineBar(
                highlight_range=(0, added_lines_ratio * bar_width),
                highlight_style="green",
                background_style="red",
                width=bar_width,
            ),
            f" [bold red]-{self.line_removals}",
        )

        bar_hpad = len(str(self.line_additions)) + len(str(self.line_removals)) + 4
        yield Align.center(line_changes_summary, width=bar_width + bar_hpad)
        yield Segment.line()


class RemovedFileBody:
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(characters="╲", style="hatched")
        yield Rule(" [red]File was removed ", characters="╲", style="hatched")
        yield Rule(characters="╲", style="hatched")
        yield Rule(style="border", characters="▔")


@dataclass
class BinaryFileBody:
    size_in_bytes: Optional[int] = None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(characters="╲", style="hatched")
        size_note = (
            f" · {self.size_in_bytes} bytes"
            if self.size_in_bytes is not None
            else ""
        )
        yield Rule(
            Text(f" File is binary{size_note} ", style="blue"),
            characters="╲",
            style="hatched",
        )
        yield Rule(characters="╲", style="hatched")
        yield Rule(style="border", characters="▔")


_TAG_STYLE = {
    TagLevel.OK: "green",
    TagLevel.WARN: "yellow",
    TagLevel.ERROR: "bold red",
}


class PatchedFileHeader:
    def __init__(
        self,
        patch: PatchedFile,
        snapshot_tags: Sequence[SnapshotTag] = (),
    ):
        self.patch = patch
        self.snapshot_tags = tuple(snapshot_tags)
        if patch.is_rename:
            self.path_prefix = (
                f"[dim][s]{escape(Path(patch.source_file).name)}[/] → [/]"
            )
        elif patch.is_added_file:
            self.path_prefix = f"[bold green]Added [/]"
        else:
            self.path_prefix = ""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        title = (
            f"{self.path_prefix}[b]{escape(self.patch.path)}[/] "
            f"([green]{self.patch.added} additions[/], "
            f"[red]{self.patch.removed} removals[/])"
        )
        if self.snapshot_tags:
            tag_parts = "  ".join(
                f"[{_TAG_STYLE[tag.level]}]⦿ {escape(tag.text)}[/]"
                for tag in self.snapshot_tags
            )
            title = f"{title}\n{tag_parts}"
        yield Rule(
            title,
            style="border",
            characters="▁",
        )


class SnapshotErrorBody:
    """Rendered when one file's sides cannot be verified in isolation."""

    def __init__(self, message: str):
        self.message = message

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(characters="╲", style="hatched")
        yield Rule(
            Text(" snapshot consistency error — file skipped ", style="bold red"),
            characters="╲",
            style="hatched",
        )
        yield Rule(
            Text(f" {self.message} ", style="red"),
            characters="╲",
            style="hatched",
        )
        yield Rule(
            Text(
                " other files and global statistics are unaffected ",
                style="dim",
            ),
            characters="╲",
            style="hatched",
        )
        yield Rule(style="border", characters="▔")


class OnlyRenamedFileBody:
    """Represents a file that was renamed but the content was not changed."""

    def __init__(self, patch: PatchedFile):
        self.patch = patch

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(characters="╲", style="hatched")
        yield Rule(" [blue]File was only renamed ", characters="╲", style="hatched")
        yield Rule(characters="╲", style="hatched")
        yield Rule(style="border", characters="▔")
