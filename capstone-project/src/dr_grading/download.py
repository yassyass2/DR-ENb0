from __future__ import annotations

import argparse
import re
import sys
from html import unescape
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

from dr_grading.paths import (
    DEFAULT_MODEL_PREPROCESSED_ARCHIVE,
    DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE,
)


GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "Mozilla/5.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the preprocessed .npz dataset archive from Google Drive."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE,
        help=(
            "Google Drive share URL or Google Drive file ID for the .npz archive. "
            "Defaults to the repository's configured dataset link."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PREPROCESSED_ARCHIVE,
        help=(
            "Destination path for the downloaded archive. "
            f"Default: {DEFAULT_MODEL_PREPROCESSED_ARCHIVE}"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    return parser


def extract_file_id(source: str) -> str:
    stripped = source.strip()
    if not stripped:
        raise ValueError("Google Drive source must not be empty.")

    if "://" not in stripped and "/" not in stripped and "?" not in stripped:
        return stripped

    parsed = urlparse(stripped)
    query_id = parse_qs(parsed.query).get("id")
    if query_id:
        return query_id[0]

    patterns = (
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1)

    raise ValueError(
        "Could not extract a Google Drive file ID from the provided source. "
        "Pass either a full share URL or the raw file ID."
    )


def build_download_url(file_id: str, confirm_token: str | None = None) -> str:
    params = {"export": "download", "id": file_id}
    if confirm_token is not None:
        params["confirm"] = confirm_token
    return f"{GOOGLE_DRIVE_DOWNLOAD_URL}?{urlencode(params)}"


def response_is_download(response) -> bool:
    content_disposition = response.headers.get("Content-Disposition", "").lower()
    content_type = response.headers.get("Content-Type", "").lower()
    if "attachment" in content_disposition:
        return True
    return not content_type.startswith("text/html")


def extract_confirm_url(html: str, file_id: str) -> str | None:
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"',
        html,
        flags=re.IGNORECASE,
    )
    if form_match:
        action = unescape(form_match.group(1))
        if action.startswith("/"):
            action = f"https://drive.google.com{action}"
        inputs = dict(
            re.findall(
                r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
                html,
                flags=re.IGNORECASE,
            )
        )
        if inputs:
            return f"{action}?{urlencode(inputs)}"

    href_match = re.search(r'href="([^"]*confirm=[^"]+)"', html, flags=re.IGNORECASE)
    if href_match:
        href = unescape(href_match.group(1)).replace("&amp;", "&")
        if href.startswith("/"):
            return f"https://drive.google.com{href}"
        return href

    token_match = re.search(r"confirm=([0-9A-Za-z_-]+)", html)
    if token_match:
        return build_download_url(file_id, confirm_token=token_match.group(1))

    return None


def stream_response_to_file(response, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    bytes_written = 0

    try:
        with temp_path.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    temp_path.replace(output_path)
    return bytes_written


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def download_google_drive_file(file_id: str, output_path: Path, force: bool) -> Path:
    output_path = output_path.resolve()
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Destination already exists: {output_path}. Use --force to overwrite it."
        )

    cookie_jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    request = Request(build_download_url(file_id), headers={"User-Agent": USER_AGENT})

    with opener.open(request) as response:
        if response_is_download(response):
            bytes_written = stream_response_to_file(response, output_path)
            print(f"Downloaded {format_bytes(bytes_written)} to {output_path}")
            return output_path

        html = response.read().decode("utf-8", errors="replace")

    confirm_url = extract_confirm_url(html, file_id)
    if confirm_url is None:
        raise RuntimeError(
            "Google Drive returned an HTML page but no download confirmation link was found. "
            "Check that the file is shared correctly and that the link points to a file."
        )

    confirm_request = Request(confirm_url, headers={"User-Agent": USER_AGENT})
    with opener.open(confirm_request) as response:
        if not response_is_download(response):
            error_preview = response.read(500).decode("utf-8", errors="replace")
            raise RuntimeError(
                "Google Drive did not return the file contents after confirmation. "
                f"Response preview: {error_preview}"
            )
        bytes_written = stream_response_to_file(response, output_path)

    print(f"Downloaded {format_bytes(bytes_written)} to {output_path}")
    return output_path


def download_google_drive_source(source: str, output_path: Path, force: bool = False) -> Path:
    file_id = extract_file_id(source)
    return download_google_drive_file(file_id=file_id, output_path=output_path, force=force)


def ensure_preprocessed_archive(
    data_path: Path,
    download_source: str | None = DEFAULT_MODEL_PREPROCESSED_DOWNLOAD_SOURCE,
) -> Path:
    data_path = Path(data_path)
    if data_path.exists():
        return data_path.resolve()

    if data_path.suffix != ".npz":
        raise FileNotFoundError(
            f"Preprocessed data source does not exist: {data_path}. "
            "Automatic download is only supported for the .npz archive."
        )

    if not download_source:
        raise FileNotFoundError(
            f"Preprocessed archive not found: {data_path}. "
            "No Google Drive download source is configured."
        )

    print(f"Preprocessed archive not found at {data_path}.")
    print("Downloading it from Google Drive before running inference...")
    return download_google_drive_source(
        source=download_source,
        output_path=data_path,
        force=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        download_google_drive_source(
            source=args.source,
            output_path=args.output,
            force=args.force,
        )
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
