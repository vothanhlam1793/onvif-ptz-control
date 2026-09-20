"""MinIO S3 Object Storage integration for Spatial Memory PTZ Agent (minio.nvlit.asia)."""

from __future__ import annotations

import io
import mimetypes
import os
import uuid
import logging
from typing import BinaryIO
import boto3
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "https://minio.nvlit.asia")
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "https://minio.nvlit.asia").rstrip("/")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "7c70570adb7108f7289d1380103ecb4d")
MINIO_DEFAULT_BUCKET = os.getenv("MINIO_DEFAULT_BUCKET", "creta-warranty")


def get_s3_client():
    """Create S3 boto3 client configured for MinIO Hub."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_file_bytes(
    file_bytes: bytes | BinaryIO,
    filename: str,
    bucket: str | None = None,
    prefix: str = "spatial_ptz",
    content_type: str | None = None,
) -> str:
    """Upload bytes or file-like object to MinIO and return the public HTTPS URL."""
    target_bucket = bucket or MINIO_DEFAULT_BUCKET
    s3 = get_s3_client()

    if not content_type:
        content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"

    unique_key = f"{prefix.strip('/')}/{uuid.uuid4().hex[:8]}_{filename.replace(' ', '_')}"
    if isinstance(file_bytes, bytes):
        body = io.BytesIO(file_bytes)
    else:
        body = file_bytes

    try:
        s3.upload_fileobj(
            body,
            target_bucket,
            unique_key,
            ExtraArgs={"ContentType": content_type},
        )
        url = f"{MINIO_PUBLIC_URL}/{target_bucket}/{unique_key}"
        return url
    except Exception as e:
        logger.warning(f"Failed to upload to MinIO: {e}")
        return ""


def upload_spatial_frame(
    image_bytes: bytes,
    cell_id: str,
    camera_key: str = "uniarch_uho_s2e",
    bucket: str | None = None,
) -> str:
    """Convenience helper to upload 3D grid scan frames."""
    prefix = f"spatial_ptz/{camera_key}/frames"
    return upload_file_bytes(
        file_bytes=image_bytes,
        filename=f"{cell_id}.jpg",
        bucket=bucket or MINIO_DEFAULT_BUCKET,
        prefix=prefix,
    )


def upload_verified_image(
    image_bytes: bytes,
    target_label: str,
    camera_key: str = "uniarch_uho_s2e",
    bucket: str | None = None,
) -> str:
    """Convenience helper to upload verified target images."""
    prefix = f"spatial_ptz/{camera_key}/verified"
    clean_label = target_label.replace(" ", "_").lower()
    return upload_file_bytes(
        file_bytes=image_bytes,
        filename=f"verify_{clean_label}.jpg",
        bucket=bucket or MINIO_DEFAULT_BUCKET,
        prefix=prefix,
    )
