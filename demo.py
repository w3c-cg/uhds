"""
UHDS (Universal Human Data Stream) - Production Implementation
Fixed for Pydantic V2 and datetime serialization.
"""
import asyncio
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod
from decimal import Decimal
import hashlib
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.json import pydantic_encoder
from contextlib import asynccontextmanager
import aiohttp
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, JSON, DateTime, Float, Integer
import numpy as np
from dataclasses_json import dataclass_json, config
import msgpack

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uhds")

# ============ CORE MODELS ============

class DataSourceType(str, Enum):
    """Types of human data sources"""
    WEARABLE = "wearable"
    MEDICAL_DEVICE = "medical_device"
    MOBILE_APP = "mobile_app"
    EHR_SYSTEM = "ehr_system"
    ENVIRONMENTAL = "environmental"
    BEHAVIORAL = "behavioral"
    COGNITIVE = "cognitive"

class DataQualityLevel(str, Enum):
    """Data quality assessment levels"""
    RAW = "raw"
    CLEANED = "cleaned"
    VALIDATED = "validated"
    CLINICAL_GRADE = "clinical_grade"
    RESEARCH_GRADE = "research_grade"

class PrivacyLevel(str, Enum):
    """Privacy and consent levels"""
    ANONYMIZED = "anonymized"
    PSEUDONYMIZED = "pseudonymized"
    IDENTIFIED = "identified"
    SENSITIVE = "sensitive"

@dataclass_json
@dataclass
class DataStreamMetadata:
    """Metadata for a UHDS data stream"""
    # REQUIRED FIELDS (no defaults) must come first
    source_type: DataSourceType
    source_id: str
    subject_id: str  # Pseudonymized/consented identifier
    
    # OPTIONAL FIELDS (with defaults) come after
    stream_id: str = field(default_factory=lambda: f"stream_{uuid.uuid4().hex[:8]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    privacy_level: PrivacyLevel = PrivacyLevel.PSEUDONYMIZED
    quality_level: DataQualityLevel = DataQualityLevel.RAW
    schema_version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    geo_region: Optional[str] = None
    timezone: str = "UTC"
    
    def to_json(self) -> str:
        """Convert metadata to JSON string"""
        return json.dumps(asdict(self), default=str, indent=2)
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper datetime handling"""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result

class UHDSDataPoint(BaseModel):
    """Base model for all UHDS data points"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    value: Union[float, int, str, Dict, List]
    unit: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Pydantic V2 config
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        },
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware"""
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Override dict() to ensure datetime serialization"""
        result = super().model_dump(**kwargs)
        # Ensure datetime is converted to string
        if isinstance(result.get('timestamp'), datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        return result
    
    def json(self, **kwargs) -> str:
        """Return JSON representation"""
        return super().model_dump_json(**kwargs)

# ============ DOMAIN-SPECIFIC MODELS ============

class VitalSign(BaseModel):
    """Vital signs data model"""
    heart_rate: Optional[float] = Field(None, ge=0, le=300, description="BPM")
    systolic_bp: Optional[float] = Field(None, ge=0, le=300, description="mmHg")
    diastolic_bp: Optional[float] = Field(None, ge=0, le=300, description="mmHg")
    spo2: Optional[float] = Field(None, ge=0, le=100, description="Percentage")
    respiratory_rate: Optional[float] = Field(None, ge=0, le=100, description="Breaths/min")
    body_temperature: Optional[float] = Field(None, ge=20, le=45, description="Celsius")
    
    @field_validator('spo2')
    @classmethod
    def validate_spo2(cls, v: Optional[float]) -> Optional[float]:
        """Validate SpO2 values"""
        if v is not None and (v < 70 or v > 100):
            logger.warning(f"Spo2 value {v} outside typical range (70-100%)")
        return v
    
    model_config = ConfigDict(validate_assignment=True)

class ActivityData(BaseModel):
    """Physical activity data model"""
    steps: int = Field(ge=0, default=0)
    distance_meters: float = Field(ge=0, default=0.0)
    calories_burned: float = Field(ge=0, default=0.0)
    active_minutes: int = Field(ge=0, default=0)
    activity_type: str = "walking"
    intensity: str = "moderate"
    
    model_config = ConfigDict(validate_assignment=True)

class SleepStage(str, Enum):
    """Sleep stage enumeration"""
    AWAKE = "awake"
    LIGHT = "light"
    DEEP = "deep"
    REM = "rem"

class SleepData(BaseModel):
    """Sleep data model"""
    start_time: datetime
    end_time: datetime
    total_duration_minutes: float
    sleep_stages: List[Dict[str, Any]] = Field(default_factory=list)
    sleep_score: Optional[float] = Field(None, ge=0, le=100)
    awakenings: int = Field(ge=0, default=0)
    
    @field_validator('end_time')
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        """Validate end time is after start time"""
        values = info.data
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError("End time must be after start time")
        return v
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat()
        }
    )

# ============ STREAM PROCESSORS ============

class StreamProcessor(ABC):
    """Abstract base class for stream processors"""
    
    @abstractmethod
    async def process(self, data_point: UHDSDataPoint) -> UHDSDataPoint:
        """Process a single data point"""
        pass
    
    @abstractmethod
    async def validate(self, data_point: UHDSDataPoint) -> bool:
        """Validate a data point"""
        pass

class QualityEnhancer(StreamProcessor):
    """Enhance data quality through validation and cleaning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.rules = config.get("validation_rules", {})
        self.outlier_threshold = config.get("outlier_threshold", 3.0)
    
    async def process(self, data_point: UHDSDataPoint) -> UHDSDataPoint:
        """Clean and validate data point"""
        # Apply outlier detection
        if isinstance(data_point.value, (int, float)):
            cleaned_value = self._remove_outliers(data_point.value)
            data_point.value = cleaned_value
        
        # Add quality metadata
        data_point.metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
        data_point.metadata["quality_processor"] = self.__class__.__name__
        
        return data_point
    
    def _remove_outliers(self, value: float) -> float:
        """Simple outlier detection"""
        # This is a simplified example
        return value
    
    async def validate(self, data_point: UHDSDataPoint) -> bool:
        """Validate data point against rules"""
        try:
            # Check timestamp is not in future
            if data_point.timestamp > datetime.now(timezone.utc):
                logger.warning(f"Future timestamp detected: {data_point.timestamp}")
                return False
            
            # Check confidence score
            if data_point.confidence < 0.3:
                logger.warning(f"Low confidence score: {data_point.confidence}")
                return False
            
            # Check against validation rules
            if isinstance(data_point.value, (int, float)):
                if "max_heart_rate" in self.rules and data_point.unit == "bpm":
                    if data_point.value > self.rules["max_heart_rate"]:
                        logger.warning(f"Heart rate exceeds maximum: {data_point.value}")
                        return False
                
                if "min_heart_rate" in self.rules and data_point.unit == "bpm":
                    if data_point.value < self.rules["min_heart_rate"]:
                        logger.warning(f"Heart rate below minimum: {data_point.value}")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

class PrivacyFilter(StreamProcessor):
    """Apply privacy-preserving transformations"""
    
    def __init__(self, privacy_level: PrivacyLevel):
        self.privacy_level = privacy_level
    
    async def process(self, data_point: UHDSDataPoint) -> UHDSDataPoint:
        """Apply privacy transformations"""
        if self.privacy_level == PrivacyLevel.ANONYMIZED:
            # Remove or hash identifiers
            if "device_id" in data_point.metadata:
                data_point.metadata["device_id"] = self._hash_id(
                    data_point.metadata["device_id"]
                )
            if "subject_id" in data_point.metadata:
                data_point.metadata["subject_id"] = self._hash_id(
                    data_point.metadata["subject_id"]
                )
        
        data_point.metadata["privacy_applied"] = self.privacy_level.value
        data_point.metadata["privacy_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return data_point
    
    def _hash_id(self, identifier: str) -> str:
        """Hash identifier for anonymization"""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    async def validate(self, data_point: UHDSDataPoint) -> bool:
        """Validate privacy compliance"""
        # Check if sensitive data is properly protected
        sensitive_fields = ["ssn", "email", "phone", "address", "name", "birthdate"]
        data_str = str(data_point.value).lower() + " " + str(data_point.metadata).lower()
        
        for field in sensitive_fields:
            if field in data_str:
                if self.privacy_level == PrivacyLevel.ANONYMIZED:
                    logger.warning(f"Sensitive data '{field}' found in anonymized stream")
                    return False
        return True

# ============ DATA PIPELINE ============

class UHDSPipeline:
    """Main UHDS processing pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processors: List[StreamProcessor] = []
        self._setup_processors()
        self._setup_storage()
    
    def _setup_processors(self):
        """Initialize all processors based on config"""
        if self.config.get("enable_quality_enhancement", True):
            self.processors.append(QualityEnhancer(self.config))
        
        privacy_level = PrivacyLevel(
            self.config.get("privacy_level", "pseudonymized")
        )
        if self.config.get("enable_privacy_filter", True):
            self.processors.append(PrivacyFilter(privacy_level))
    
    def _setup_storage(self):
        """Setup storage backends"""
        # In production, connect to actual databases
        self.storage_backends = {
            "redis": None,  # Would initialize redis client
            "postgres": None,  # Would initialize postgres connection
        }
    
    async def process_stream(
        self, 
        data_points: List[UHDSDataPoint],
        metadata: DataStreamMetadata
    ) -> Dict[str, Any]:
        """Process a batch of data points through the pipeline"""
        
        results = {
            "metadata": metadata.dict(),
            "processed": [],
            "failed": [],
            "statistics": {},
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "2.0.0"
        }
        
        for data_point in data_points:
            try:
                processed_point = data_point
                
                # Run through all processors
                for processor in self.processors:
                    # Validate first
                    if not await processor.validate(processed_point):
                        raise ValueError(f"Validation failed by {processor.__class__.__name__}")
                    
                    # Process
                    processed_point = await processor.process(processed_point)
                
                # Update metadata
                processed_point.metadata["pipeline_version"] = "2.0.0"
                processed_point.metadata["stream_id"] = metadata.stream_id
                processed_point.metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
                
                # Convert to dict with proper datetime handling
                point_dict = processed_point.dict()
                results["processed"].append(point_dict)
                
            except Exception as e:
                logger.error(f"Failed to process data point: {e}")
                results["failed"].append({
                    "data_point": data_point.dict(),
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Generate statistics
        results["statistics"] = self._generate_statistics(results["processed"])
        
        # Store results
        await self._store_results(results, metadata)
        
        return results
    
    def _generate_statistics(self, processed_points: List[Dict]) -> Dict:
        """Generate statistics from processed data"""
        if not processed_points:
            return {"count": 0}
        
        numeric_values = []
        for p in processed_points:
            val = p.get("value")
            if isinstance(val, (int, float)):
                numeric_values.append(val)
        
        if numeric_values:
            arr = np.array(numeric_values)
            return {
                "count": len(numeric_values),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "q1": float(np.percentile(arr, 25)),
                "q3": float(np.percentile(arr, 75))
            }
        
        return {"count": len(processed_points)}
    
    async def _store_results(self, results: Dict, metadata: DataStreamMetadata):
        """Store processed results"""
        # Create directory if it doesn't exist
        import os
        os.makedirs("data/processed", exist_ok=True)
        
        timestamp = int(time.time())
        storage_path = f"data/processed/{metadata.stream_id}_{timestamp}.json"
        try:
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results stored to {storage_path}")
            
            # Also create a summary file
            summary = {
                "stream_id": metadata.stream_id,
                "timestamp": timestamp,
                "processed_count": len(results["processed"]),
                "failed_count": len(results["failed"]),
                "file_path": storage_path
            }
            
            with open(f"data/processed/summary_{timestamp}.json", 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to store results: {e}")

# ============ REAL-WORLD EXAMPLE IMPLEMENTATIONS ============

class FitbitDataAdapter:
    """Adapter for Fitbit wearable data"""
    
    @staticmethod
    def convert_to_uhds(fitbit_data: Dict) -> List[UHDSDataPoint]:
        """Convert Fitbit API response to UHDS format"""
        uhds_points = []
        
        # Convert heart rate data
        if "activities-heart" in fitbit_data:
            for entry in fitbit_data["activities-heart"]:
                # Handle timestamp conversion
                dt_str = entry["dateTime"]
                if "Z" in dt_str:
                    dt_str = dt_str.replace("Z", "+00:00")
                
                point = UHDSDataPoint(
                    timestamp=datetime.fromisoformat(dt_str),
                    value=entry["value"].get("restingHeartRate", 0),
                    unit="bpm",
                    confidence=0.9,
                    metadata={
                        "source": "fitbit",
                        "data_type": "heart_rate",
                        "device_type": "wearable",
                        "original_format": "fitbit_api"
                    }
                )
                uhds_points.append(point)
        
        # Convert sleep data
        if "sleep" in fitbit_data:
            for sleep in fitbit_data["sleep"]:
                dt_str = sleep["startTime"]
                if "Z" in dt_str:
                    dt_str = dt_str.replace("Z", "+00:00")
                
                point = UHDSDataPoint(
                    timestamp=datetime.fromisoformat(dt_str),
                    value=sleep["minutesAsleep"],
                    unit="minutes",
                    confidence=0.8,
                    metadata={
                        "source": "fitbit",
                        "data_type": "sleep_duration",
                        "sleep_stages": sleep.get("levels", {}).get("data", []),
                        "original_format": "fitbit_api"
                    }
                )
                uhds_points.append(point)
        
        # Convert step data
        if "activities-steps" in fitbit_data:
            for entry in fitbit_data["activities-steps"]:
                dt_str = entry["dateTime"]
                if "Z" in dt_str:
                    dt_str = dt_str.replace("Z", "+00:00")
                
                point = UHDSDataPoint(
                    timestamp=datetime.fromisoformat(dt_str),
                    value=entry["value"],
                    unit="steps",
                    confidence=0.85,
                    metadata={
                        "source": "fitbit",
                        "data_type": "step_count",
                        "device_type": "wearable",
                        "original_format": "fitbit_api"
                    }
                )
                uhds_points.append(point)
        
        return uhds_points

class AppleHealthAdapter:
    """Adapter for Apple HealthKit data"""
    
    @staticmethod
    def convert_to_uhds(health_data: Dict) -> List[UHDSDataPoint]:
        """Convert Apple Health data to UHDS format"""
        uhds_points = []
        
        # Convert step count
        if "step_count" in health_data:
            step_data = health_data["step_count"]
            dt_str = step_data["timestamp"]
            if "Z" in dt_str:
                dt_str = dt_str.replace("Z", "+00:00")
            
            point = UHDSDataPoint(
                timestamp=datetime.fromisoformat(dt_str),
                value=step_data["value"],
                unit="count",
                confidence=0.95,
                metadata={
                    "source": "apple_health",
                    "data_type": "steps",
                    "device_uuid": health_data.get("device_uuid", ""),
                    "original_format": "apple_health_export"
                }
            )
            uhds_points.append(point)
        
        # Convert heart rate data
        if "heart_rate" in health_data:
            hr_data = health_data["heart_rate"]
            dt_str = hr_data["timestamp"]
            if "Z" in dt_str:
                dt_str = dt_str.replace("Z", "+00:00")
            
            point = UHDSDataPoint(
                timestamp=datetime.fromisoformat(dt_str),
                value=hr_data["value"],
                unit="bpm",
                confidence=0.92,
                metadata={
                    "source": "apple_health",
                    "data_type": "heart_rate",
                    "original_format": "apple_health_export"
                }
            )
            uhds_points.append(point)
        
        return uhds_points

# ============ JSON UTILITIES ============

class UHDSJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for UHDS objects"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            return obj.dict()
        if hasattr(obj, 'dict'):
            return obj.dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """Safely convert data to JSON string"""
    return json.dumps(data, cls=UHDSJsonEncoder, indent=indent, ensure_ascii=False)

# ============ DEMO FUNCTIONS ============

async def run_demo():
    """Run a comprehensive demonstration of UHDS"""
    print("=" * 60)
    print("UHDS DEMONSTRATION - Pydantic V2")
    print("=" * 60)
    
    # 1. Create configuration
    config = {
        "enable_quality_enhancement": True,
        "enable_privacy_filter": True,
        "privacy_level": "pseudonymized",
        "validation_rules": {
            "max_heart_rate": 220,
            "min_heart_rate": 30,
            "max_spo2": 100,
            "min_spo2": 70
        }
    }
    
    # 2. Create UHDS pipeline
    pipeline = UHDSPipeline(config)
    
    # 3. Create stream metadata
    metadata = DataStreamMetadata(
        source_type=DataSourceType.WEARABLE,
        source_id="apple_watch_series_8",
        subject_id="user_001",
        tags=["demo", "wearable", "health_tracking"],
        privacy_level=PrivacyLevel.PSEUDONYMIZED,
        quality_level=DataQualityLevel.VALIDATED
    )
    
    # 4. Create comprehensive sample data
    print("\n1. Creating sample data points...")
    sample_data = [
        UHDSDataPoint(
            timestamp=datetime.now(timezone.utc),
            value=72.5,
            unit="bpm",
            confidence=0.95,
            metadata={
                "source": "apple_watch",
                "reading_type": "resting",
                "device_id": "AW123456",
                "subject_name": "John Doe"
            }
        ),
        UHDSDataPoint(
            timestamp=datetime.now(timezone.utc),
            value=98.2,
            unit="%",
            confidence=0.92,
            metadata={
                "source": "apple_watch",
                "reading_type": "resting",
                "device_id": "AW123456",
                "subject_name": "John Doe"
            }
        ),
        UHDSDataPoint(
            timestamp=datetime.now(timezone.utc),
            value=36.8,
            unit="°C",
            confidence=0.98,
            metadata={
                "source": "smart_thermometer",
                "reading_type": "body",
                "device_id": "THERMO001",
                "subject_name": "John Doe"
            }
        ),
        UHDSDataPoint(
            timestamp=datetime.now(timezone.utc),
            value=12000,
            unit="steps",
            confidence=0.88,
            metadata={
                "source": "apple_watch",
                "reading_type": "daily_total",
                "device_id": "AW123456",
                "subject_name": "John Doe"
            }
        ),
        UHDSDataPoint(
            timestamp=datetime.now(timezone.utc),
            value=7.5,  # hours
            unit="hours",
            confidence=0.75,
            metadata={
                "source": "sleep_tracker",
                "reading_type": "sleep_duration",
                "device_id": "SLEEP001",
                "subject_name": "John Doe"
            }
        )
    ]
    
    print(f"   Created {len(sample_data)} sample data points")
    
    # 5. Process the data
    print("\n2. Processing data through UHDS pipeline...")
    results = await pipeline.process_stream(sample_data, metadata)
    
    # 6. Display results
    print("\n3. Results Summary:")
    print("-" * 40)
    print(f"✓ Processed: {len(results['processed'])} data points")
    print(f"✗ Failed: {len(results['failed'])} data points")
    
    stats = results['statistics']
    if stats['count'] > 0:
        print(f"📊 Statistics:")
        print(f"   Count: {stats['count']}")
        print(f"   Mean: {stats['mean']:.2f}")
        print(f"   Min: {stats['min']:.2f}")
        print(f"   Max: {stats['max']:.2f}")
        print(f"   Std Dev: {stats['std']:.2f}")
    
    # 7. Show metadata
    print(f"\n4. Stream Metadata:")
    print("-" * 40)
    print(f"   Stream ID: {metadata.stream_id}")
    print(f"   Source: {metadata.source_type.value}")
    print(f"   Device: {metadata.source_id}")
    print(f"   Subject: {metadata.subject_id}")
    print(f"   Privacy: {metadata.privacy_level.value}")
    print(f"   Quality: {metadata.quality_level.value}")
    print(f"   Created: {metadata.created_at.isoformat()}")
    
    # 8. Show first processed point
    if results['processed']:
        print(f"\n5. First Processed Data Point:")
        print("-" * 40)
        first_point = results['processed'][0]
        print(f"   Value: {first_point['value']} {first_point['unit']}")
        print(f"   Confidence: {first_point['confidence']}")
        print(f"   Timestamp: {first_point['timestamp']}")
        print(f"   Metadata keys: {list(first_point['metadata'].keys())}")
        
        # Show privacy transformations
        if 'privacy_applied' in first_point['metadata']:
            print(f"   Privacy: {first_point['metadata']['privacy_applied']}")
    
    # 9. Demonstrate adapters
    print(f"\n6. Adapter Demonstration:")
    print("-" * 40)
    
    # Fitbit example
    fitbit_sample = {
        "activities-heart": [
            {
                "dateTime": "2024-01-15T08:00:00Z",
                "value": {"restingHeartRate": 65}
            },
            {
                "dateTime": "2024-01-15T12:00:00Z", 
                "value": {"restingHeartRate": 72}
            }
        ],
        "activities-steps": [
            {
                "dateTime": "2024-01-15T00:00:00Z",
                "value": 12543
            }
        ]
    }
    
    fitbit_points = FitbitDataAdapter.convert_to_uhds(fitbit_sample)
    print(f"   Fitbit Adapter: Converted {len(fitbit_points)} data points")
    
    # Process Fitbit data
    fitbit_metadata = DataStreamMetadata(
        source_type=DataSourceType.WEARABLE,
        source_id="fitbit_charge5",
        subject_id="fitbit_user_001"
    )
    
    fitbit_results = await pipeline.process_stream(fitbit_points, fitbit_metadata)
    print(f"   Processed Fitbit data: {len(fitbit_results['processed'])} successful")
    
    # 10. Show file output
    print(f"\n7. Output Files:")
    print("-" * 40)
    import os
    if os.path.exists("data/processed"):
        files = os.listdir("data/processed")
        json_files = [f for f in files if f.endswith('.json')]
        print(f"   Created {len(json_files)} JSON files in data/processed/")
        if json_files:
            print(f"   Latest: {json_files[-1]}")
    
    return results

def create_simple_demo():
    """Simple synchronous demo for quick testing"""
    print("UHDS Simple Demo - Pydantic V2")
    print("=" * 40)
    
    # Create metadata
    metadata = DataStreamMetadata(
        source_type=DataSourceType.MEDICAL_DEVICE,
        source_id="ecg_machine_001",
        subject_id="patient_123"
    )
    
    print(f"✓ Created stream: {metadata.stream_id}")
    print(f"✓ Source: {metadata.source_type.value}")
    print(f"✓ Subject: {metadata.subject_id}")
    print(f"✓ Created: {metadata.created_at.isoformat()}")
    
    # Create a simple data point
    data_point = UHDSDataPoint(
        value=75.0,
        unit="bpm",
        metadata={"device": "Polar_H10", "reading": "resting_hr", "subject": "Patient A"}
    )
    
    print(f"\n✓ Created data point:")
    print(f"  Value: {data_point.value} {data_point.unit}")
    print(f"  Timestamp: {data_point.timestamp.isoformat()}")
    print(f"  Confidence: {data_point.confidence}")
    
    # Show JSON representation
    print(f"\n✓ JSON Representation:")
    print(safe_json_dumps(data_point.dict(), indent=2))
    
    # Test privacy filter
    print(f"\n✓ Privacy Filter Test:")
    privacy_filter = PrivacyFilter(PrivacyLevel.ANONYMIZED)
    
    # Create a copy for privacy testing
    test_point = UHDSDataPoint(
        value=80.0,
        unit="bpm",
        metadata={"device_id": "DEVICE123", "subject_id": "PATIENT456", "name": "John Doe"}
    )
    
    print(f"  Before: device_id = {test_point.metadata.get('device_id')}")
    print(f"  Before: subject_id = {test_point.metadata.get('subject_id')}")
    
    # Apply privacy (in async context)
    async def apply_privacy():
        return await privacy_filter.process(test_point)
    
    # Run sync
    import asyncio
    processed_point = asyncio.run(apply_privacy())
    
    print(f"  After: device_id = {processed_point.metadata.get('device_id')}")
    print(f"  After: subject_id = {processed_point.metadata.get('subject_id')}")
    print(f"  Privacy applied: {processed_point.metadata.get('privacy_applied')}")
    
    return metadata, data_point, processed_point

# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    import sys
    import os
    
    # Create output directory
    os.makedirs("data/processed", exist_ok=True)
    
    try:
        # Check if we're in async context
        if sys.platform == "win32":
            # Windows requires special event loop policy
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run simple demo first
        print("\n" + "="*60)
        print("RUNNING SIMPLE SYNC DEMO")
        print("="*60)
        metadata, data_point, processed_point = create_simple_demo()
        
        # Ask if user wants to run async demo
        print("\n" + "="*60)
        response = input("\nRun full async demo? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\n" + "="*60)
            print("RUNNING FULL ASYNC DEMO")
            print("="*60)
            results = asyncio.run(run_demo())
            
            # Save results to file
            os.makedirs("output", exist_ok=True)
            output_file = "output/demo_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(safe_json_dumps(results, indent=2))
            
            print(f"\n✓ Results saved to '{output_file}'")
            print(f"✓ Demo completed successfully!")
            
            # Show quick stats
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✓ Output file size: {file_size / 1024:.2f} KB")
        
        else:
            print("\n✓ Simple demo completed successfully!")
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting tips:")
        print("1. Install required packages: pip install pydantic numpy dataclasses-json")
        print("2. Check Python version (3.9+ required for Pydantic V2)")
        print("3. If async issues, try: pip install nest-asyncio")
        print("4. For Windows, ensure event loop policy is set")
        
        # Try to install missing packages
        try:
            import pydantic
            print(f"✓ Pydantic version: {pydantic.__version__}")
        except ImportError:
            print("❌ Pydantic not installed. Run: pip install pydantic")
        
        try:
            import numpy
            print(f"✓ NumPy version: {numpy.__version__}")
        except ImportError:
            print("❌ NumPy not installed. Run: pip install numpy")