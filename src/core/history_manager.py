"""
History Manager for InfraSight
Handles SQLite database operations using SQLAlchemy to persist analysis results.
"""
import os
import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

Base = declarative_base()

class AnalysisHistory(Base):
    __tablename__ = 'analysis_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    image_name = Column(String(255))
    image_path = Column(String(511))
    
    # Metrics
    area_cm2 = Column(Float)
    avg_depth_cm = Column(Float)
    volume_cm3 = Column(Float)
    
    # Analysis
    severity_level = Column(String(50))
    severity_score = Column(Float)
    repair_method = Column(String(100))
    repair_cost_idr = Column(Float)
    repair_material_kg = Column(Float)
    
    # Location
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # JSON Data (for 3D and other details)
    metadata_json = Column(Text, nullable=True)

class HistoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # Resolve relative to project root
            root_dir = Path(__file__).parent.parent.parent.absolute()
            self.db_path = root_dir / "data" / "infrasight.db"
        else:
            self.db_path = Path(db_path)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        db_url = f"sqlite:///{self.db_path.absolute()}"
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_analysis(self, analysis_data):
        """
        Save analysis result to database.
        analysis_data: dict containing keys matching AnalysisHistory columns
        """
        session = self.Session()
        try:
            history_entry = AnalysisHistory(
                image_name=analysis_data.get('image_name'),
                image_path=analysis_data.get('image_path'),
                area_cm2=analysis_data.get('area_cm2'),
                avg_depth_cm=analysis_data.get('avg_depth_cm'),
                volume_cm3=analysis_data.get('volume_cm3'),
                severity_level=analysis_data.get('severity_level'),
                severity_score=analysis_data.get('severity_score'),
                repair_method=analysis_data.get('repair_method'),
                repair_cost_idr=analysis_data.get('repair_cost_idr'),
                repair_material_kg=analysis_data.get('repair_material_kg'),
                latitude=analysis_data.get('latitude'),
                longitude=analysis_data.get('longitude'),
                metadata_json=analysis_data.get('metadata_json')
            )
            session.add(history_entry)
            session.commit()
            return history_entry.id
        except Exception as e:
            session.rollback()
            print(f"Error saving history: {e}")
            return None
        finally:
            session.close()

    def get_all_history(self, limit=100):
        """Fetch all history entries sorted by timestamp desc"""
        session = self.Session()
        try:
            return session.query(AnalysisHistory).order_by(AnalysisHistory.timestamp.desc()).limit(limit).all()
        finally:
            session.close()

    def get_entry_by_id(self, entry_id):
        """Fetch a specific entry by ID"""
        session = self.Session()
        try:
            return session.query(AnalysisHistory).filter(AnalysisHistory.id == entry_id).first()
        finally:
            session.close()

    def delete_entry(self, entry_id):
        """Delete an entry by ID"""
        session = self.Session()
        try:
            entry = session.query(AnalysisHistory).filter(AnalysisHistory.id == entry_id).first()
            if entry:
                session.delete(entry)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting entry: {e}")
            return False
        finally:
            session.close()
