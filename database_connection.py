import os
import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_database_connection():
    """
    Establishes a connection to the PostgreSQL database.
    Uses st.cache_resource to maintain the connection across reruns.
    """
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        st.error("Database credentials are missing. Please check your .env file.")
        return None

    try:
        # Construct the connection string
        db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_timeout=30,
            future=True,
            connect_args={
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )
        return engine
    except Exception:
        st.error("Something went wrong. Please refresh the page or try again.")
        st.stop()
