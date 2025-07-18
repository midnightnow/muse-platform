"""
Database Configuration and Initialization for MUSE Platform

This module provides database connection management, session handling,
and initialization for the MUSE Computational Platonism platform.
"""

import os
import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from urllib.parse import urlparse
import time

# Import all models to ensure they are registered
from muse.models.community import (
    Base, UserProfile, FrequencySignature, CommunityCreation,
    Comment, Like, Follow, CollaborativeSession, SessionParticipant,
    ResonanceCache, CommunityAnalytics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DEFAULT_DATABASE_URL = "sqlite:///./muse_platform.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

# Connection pool settings
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# SQLite specific settings
SQLITE_TIMEOUT = int(os.getenv("SQLITE_TIMEOUT", "20"))

class DatabaseManager:
    """
    Database connection and session management for MUSE platform
    """
    
    def __init__(self, database_url: str = DATABASE_URL):
        """
        Initialize database manager
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()
        self._setup_session_factory()
    
    def _setup_engine(self):
        """Setup SQLAlchemy engine with appropriate configuration"""
        parsed_url = urlparse(self.database_url)
        
        if parsed_url.scheme == "sqlite":
            # SQLite configuration
            self.engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": SQLITE_TIMEOUT
                },
                echo=os.getenv("DB_ECHO", "").lower() == "true"
            )
            
            # Enable foreign key constraints for SQLite
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=1000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
                
        elif parsed_url.scheme == "postgresql":
            # PostgreSQL configuration
            self.engine = create_engine(
                self.database_url,
                pool_size=POOL_SIZE,
                max_overflow=MAX_OVERFLOW,
                pool_timeout=POOL_TIMEOUT,
                pool_recycle=POOL_RECYCLE,
                pool_pre_ping=True,
                echo=os.getenv("DB_ECHO", "").lower() == "true"
            )
            
        else:
            # Default configuration for other databases
            self.engine = create_engine(
                self.database_url,
                pool_size=POOL_SIZE,
                max_overflow=MAX_OVERFLOW,
                pool_timeout=POOL_TIMEOUT,
                pool_recycle=POOL_RECYCLE,
                echo=os.getenv("DB_ECHO", "").lower() == "true"
            )
        
        logger.info(f"Database engine created for: {parsed_url.scheme}")
    
    def _setup_session_factory(self):
        """Setup session factory"""
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        logger.info("Session factory configured")
    
    def get_db(self) -> Generator[Session, None, None]:
        """
        Dependency function for FastAPI to get database session
        
        Yields:
            Database session
        """
        db = self.SessionLocal()
        try:
            yield db
        except Exception as e:
            logger.error(f"Database session error: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions
        
        Yields:
            Database session
        """
        db = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception as e:
            logger.error(f"Database session error: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise
    
    def reset_database(self):
        """Reset database by dropping and recreating tables"""
        logger.warning("Resetting database - all data will be lost!")
        self.drop_tables()
        self.create_tables()
        logger.info("Database reset completed")
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.engine.connect() as connection:
                connection.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_engine_info(self) -> dict:
        """
        Get information about the database engine
        
        Returns:
            Dictionary with engine information
        """
        return {
            "url": str(self.engine.url),
            "driver": self.engine.driver,
            "dialect": self.engine.dialect.name,
            "pool_size": getattr(self.engine.pool, 'size', 'N/A'),
            "max_overflow": getattr(self.engine.pool, 'max_overflow', 'N/A'),
            "pool_timeout": getattr(self.engine.pool, 'timeout', 'N/A'),
            "pool_recycle": getattr(self.engine.pool, 'recycle', 'N/A'),
        }
    
    def get_table_info(self) -> dict:
        """
        Get information about database tables
        
        Returns:
            Dictionary with table information
        """
        tables = {}
        
        for table_name, table in Base.metadata.tables.items():
            tables[table_name] = {
                "columns": len(table.columns),
                "indexes": len(table.indexes),
                "foreign_keys": len(table.foreign_keys),
                "primary_key": [col.name for col in table.primary_key.columns]
            }
        
        return tables
    
    def health_check(self) -> dict:
        """
        Comprehensive database health check
        
        Returns:
            Dictionary with health check results
        """
        health_data = {
            "status": "unknown",
            "timestamp": time.time(),
            "connection_test": False,
            "engine_info": {},
            "table_info": {},
            "error": None
        }
        
        try:
            # Test connection
            health_data["connection_test"] = self.test_connection()
            
            # Get engine info
            health_data["engine_info"] = self.get_engine_info()
            
            # Get table info
            health_data["table_info"] = self.get_table_info()
            
            # Overall status
            health_data["status"] = "healthy" if health_data["connection_test"] else "unhealthy"
            
        except Exception as e:
            health_data["status"] = "error"
            health_data["error"] = str(e)
            logger.error(f"Database health check failed: {e}")
        
        return health_data


# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for backward compatibility
def get_db() -> Generator[Session, None, None]:
    """Get database session (FastAPI dependency)"""
    return db_manager.get_db()

def get_session():
    """Get database session (context manager)"""
    return db_manager.get_session()

def create_tables():
    """Create all database tables"""
    db_manager.create_tables()

def drop_tables():
    """Drop all database tables"""
    db_manager.drop_tables()

def reset_database():
    """Reset database"""
    db_manager.reset_database()

def test_connection() -> bool:
    """Test database connection"""
    return db_manager.test_connection()

def health_check() -> dict:
    """Database health check"""
    return db_manager.health_check()

# Database initialization and seeding functions

def init_database(reset: bool = False):
    """
    Initialize database with tables and optional seed data
    
    Args:
        reset: Whether to reset the database before initialization
    """
    if reset:
        logger.warning("Resetting database...")
        db_manager.reset_database()
    else:
        logger.info("Creating database tables...")
        db_manager.create_tables()
    
    # Seed initial data if needed
    seed_initial_data()
    
    logger.info("Database initialization completed")

def seed_initial_data():
    """Seed database with initial data"""
    try:
        with db_manager.get_session() as db:
            # Check if we already have data
            existing_users = db.query(UserProfile).count()
            
            if existing_users == 0:
                logger.info("Seeding initial data...")
                
                # Create a sample user for testing
                sample_user = UserProfile(
                    username="muse_discoverer",
                    email="discoverer@muse.platform",
                    display_name="MUSE Discoverer",
                    bio="Exploring the mathematical realm of creative discovery",
                    primary_muse="SOPHIA",
                    secondary_muse="CALLIOPE",
                    harmonic_blend={
                        "SOPHIA": 0.4,
                        "CALLIOPE": 0.3,
                        "ERATO": 0.2,
                        "URANIA": 0.1
                    },
                    sacred_ratios={
                        "phi": 0.8,
                        "pi": 0.6,
                        "fibonacci": 0.7,
                        "e": 0.5
                    },
                    spiral_coordinates={
                        "x": 1.618,
                        "y": 3.141,
                        "z": 2.718,
                        "radius": 2.2,
                        "angle": 1.4,
                        "height": 1.7
                    },
                    preferred_forms=["sonnet", "haiku", "villanelle"],
                    favorite_themes=["wisdom", "cosmos", "beauty"],
                    discovery_style="mathematical"
                )
                
                db.add(sample_user)
                db.commit()
                
                logger.info("Initial data seeded successfully")
            else:
                logger.info(f"Database already contains {existing_users} users, skipping seed")
                
    except Exception as e:
        logger.error(f"Error seeding initial data: {e}")
        raise

# CLI interface for database management

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MUSE Database Management")
    parser.add_argument("command", choices=["init", "reset", "test", "health", "info", "seed"], 
                       help="Command to execute")
    parser.add_argument("--force", action="store_true", 
                       help="Force operation without confirmation")
    
    args = parser.parse_args()
    
    if args.command == "init":
        print("🏗️  Initializing MUSE database...")
        init_database(reset=False)
        print("✅ Database initialized successfully!")
        
    elif args.command == "reset":
        if not args.force:
            confirm = input("⚠️  This will delete all data. Are you sure? (y/N): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                exit(0)
        
        print("🔄 Resetting MUSE database...")
        init_database(reset=True)
        print("✅ Database reset completed!")
        
    elif args.command == "test":
        print("🧪 Testing database connection...")
        if test_connection():
            print("✅ Database connection successful!")
        else:
            print("❌ Database connection failed!")
            exit(1)
            
    elif args.command == "health":
        print("🏥 Checking database health...")
        health_data = health_check()
        print(f"Status: {health_data['status']}")
        print(f"Connection: {'✅' if health_data['connection_test'] else '❌'}")
        print(f"Engine: {health_data['engine_info'].get('dialect', 'unknown')}")
        print(f"Tables: {len(health_data['table_info'])}")
        
        if health_data['error']:
            print(f"Error: {health_data['error']}")
            
    elif args.command == "info":
        print("📊 Database Information:")
        engine_info = db_manager.get_engine_info()
        table_info = db_manager.get_table_info()
        
        print(f"URL: {engine_info['url']}")
        print(f"Dialect: {engine_info['dialect']}")
        print(f"Driver: {engine_info['driver']}")
        print(f"Pool Size: {engine_info['pool_size']}")
        print(f"Max Overflow: {engine_info['max_overflow']}")
        print(f"\nTables ({len(table_info)}):")
        
        for table_name, info in table_info.items():
            print(f"  {table_name}: {info['columns']} columns, {info['indexes']} indexes")
            
    elif args.command == "seed":
        print("🌱 Seeding database with initial data...")
        seed_initial_data()
        print("✅ Database seeded successfully!")
        
    print("\n🎭 MUSE Database Management Complete!")