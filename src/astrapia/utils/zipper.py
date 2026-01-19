__all__ = ["archive", "unarchive"]

import logging
import pathlib
import zipfile

import pyzipper


def archive(
    dir_to_archive: pathlib.Path,
    password: str | None = None,
    exclude_extensions: list[str] | None = None,
    include_extensions: list[str] | None = None,
) -> None:
    """Archive a directory.

    Args:
        dir_to_archive (pathlib.Path): Path to the directory to archive.
        password (str | None): Password to use for the zip file.
        exclude_extensions (list[str]): Extensions to exclude from the zip file.
        include_extensions (list[str]): Extensions to include in the zip file.
    """
    try:
        with (
            zipfile.ZipFile(dir_to_archive.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED)
            if password is None
            else pyzipper.AESZipFile(
                dir_to_archive.with_suffix(".zip"),
                "w",
                compression=pyzipper.ZIP_DEFLATED,
                encryption=pyzipper.WZ_AES,
            )
        ) as zipf:
            if password is not None:
                zipf.setpassword(password.encode())
            for f in dir_to_archive.rglob("*"):
                if exclude_extensions and f.suffix in exclude_extensions:
                    continue
                if include_extensions and f.suffix not in include_extensions:
                    continue
                zipf.write(f, f.relative_to(dir_to_archive.parent))

    except Exception as e:
        logging.error(f"archive error: {e}")


def unarchive(
    archive_file: pathlib.Path,
    password: str | None = None,
    output_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    """Unarchive a archive file.

    Args:
        archive_file (pathlib.Path): Path to the archive file to unarchive.
        password (str | None): Password to use for the zip file.
        output_dir (pathlib.Path | None): Path to the output directory to unarchive the archive file.
    """
    try:
        with (
            zipfile.ZipFile(archive_file, "r", zipfile.ZIP_DEFLATED)
            if password is None
            else pyzipper.AESZipFile(
                archive_file,
                "r",
                compression=pyzipper.ZIP_DEFLATED,
                encryption=pyzipper.WZ_AES,
            )
        ) as zipf:
            if password is not None:
                zipf.setpassword(password.encode())
            zipf.extractall(output_dir or archive_file.parent)

    except RuntimeError as e:
        logging.error(f"unarchive error: Wrong password or unsupported encryption: {e}")
        return None

    except Exception as e:
        logging.error(f"unarchive error: {e}")
        return None

    return output_dir or archive_file.parent
