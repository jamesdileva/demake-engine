"""
SQLAlchemy ORM models for the Demake Engine.
Matches the schema defined in ARCHITECTURE.md exactly.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def new_uuid() -> str:
    return str(uuid.uuid4())


class Demake(Base):
    """
    Top-level record for a single generation run.
    Created when a video is uploaded, updated as the pipeline progresses.
    """
    __tablename__ = "demakes"

    id            = Column(String, primary_key=True, default=new_uuid)
    title         = Column(String, nullable=True)
    status        = Column(String, nullable=False, default="queued")
    source_path   = Column(String, nullable=True)   # Path to uploaded MP4
    error_message = Column(Text,   nullable=True)   # Populated on failure
    created_at    = Column(DateTime, default=datetime.utcnow)
    completed_at  = Column(DateTime, nullable=True)

    # Relationships
    game_config = relationship("GameConfig", back_populates="demake", uselist=False)
    assets      = relationship("Asset",      back_populates="demake")

    def to_status_dict(self, stage: int = 0, total: int = 8, pct: int = 0, msg: str = "") -> dict:
        return {
            "demake_id":    self.id,
            "status":       self.status,
            "stage":        stage,
            "total_stages": total,
            "progress_pct": pct,
            "message":      msg,
            "error":        self.error_message,
        }


class GameConfig(Base):
    """
    Generated game configuration extracted from the trailer.
    One-to-one with a Demake.
    """
    __tablename__ = "game_configs"

    id             = Column(String, primary_key=True, default=new_uuid)
    demake_id      = Column(String, ForeignKey("demakes.id"), nullable=False)
    template_id    = Column(String, nullable=True)   # e.g. "wave_shooter"
    genre          = Column(String, nullable=True)   # Human-readable
    color_palette  = Column(Text,   nullable=True)   # JSON array of hex codes
    mechanics      = Column(Text,   nullable=True)   # JSON object
    hud_elements   = Column(Text,   nullable=True)   # JSON array
    vlm_raw_output = Column(Text,   nullable=True)   # Raw VLM response for debugging

    demake = relationship("Demake", back_populates="game_config")


class Asset(Base):
    """
    A single generated file (sprite, tile, audio) linked to a Demake.
    """
    __tablename__ = "assets"

    id               = Column(String,  primary_key=True, default=new_uuid)
    demake_id        = Column(String,  ForeignKey("demakes.id"), nullable=False)
    asset_type       = Column(String,  nullable=False)   # e.g. "sprite_player"
    slot_name        = Column(String,  nullable=True)    # Template slot this fills
    file_path        = Column(String,  nullable=True)    # Local path to file
    frame_count      = Column(Integer, nullable=True)
    frame_width      = Column(Integer, nullable=True)
    animation_states = Column(Text,    nullable=True)    # JSON array
    created_at       = Column(DateTime, default=datetime.utcnow)

    demake = relationship("Demake", back_populates="assets")


class AssetCache(Base):
    """
    Deduplication cache — if two trailers produce the same sprite description,
    skip regeneration and reuse the existing file.
    """
    __tablename__ = "asset_cache"

    id               = Column(String, primary_key=True, default=new_uuid)
    description_hash = Column(String, unique=True, nullable=False)  # SHA256
    file_path        = Column(String, nullable=False)
    created_at       = Column(DateTime, default=datetime.utcnow)