"""
Test script to verify Supabase connection setup.
Run this to check if your Supabase configuration is working correctly.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment_variables():
    """Check if required environment variables are set."""
    print("=" * 60)
    print("Testing Environment Variables")
    print("=" * 60)
    
    # Try to load from .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("[OK] Loaded .env file")
    except ImportError:
        print("[WARN] python-dotenv not installed (optional)")
        print("  Install with: pip install python-dotenv")
    
    # Check for connection string
    database_url = os.getenv('SUPABASE_DATABASE_URL')
    if database_url:
        # Mask password in output
        masked_url = database_url.split('@')[0].split(':')
        if len(masked_url) >= 3:
            masked_url[2] = '***'
            masked = ':'.join(masked_url) + '@' + '@'.join(database_url.split('@')[1:])
        else:
            masked = database_url
        print(f"[OK] SUPABASE_DATABASE_URL is set: {masked}")
        return True
    else:
        print("[FAIL] SUPABASE_DATABASE_URL not found")
        
        # Check individual components
        supabase_url = os.getenv('SUPABASE_URL')
        db_password = os.getenv('SUPABASE_DB_PASSWORD')
        
        if supabase_url:
            print(f"[OK] SUPABASE_URL is set: {supabase_url}")
        else:
            print("[FAIL] SUPABASE_URL not found")
            
        if db_password:
            print(f"[OK] SUPABASE_DB_PASSWORD is set: {'*' * len(db_password)}")
        else:
            print("[FAIL] SUPABASE_DB_PASSWORD not found")
            
        if not supabase_url or not db_password:
            print("\n[WARN] Either SUPABASE_DATABASE_URL or both SUPABASE_URL and SUPABASE_DB_PASSWORD must be set")
            return False
    
    return True

def test_imports():
    """Check if required packages are installed."""
    print("\n" + "=" * 60)
    print("Testing Required Packages")
    print("=" * 60)
    
    required_packages = {
        'sqlalchemy': 'SQLAlchemy',
        'psycopg2': 'psycopg2-binary',
    }
    
    all_installed = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[FAIL] {package} is NOT installed")
            print(f"  Install with: pip install {package}")
            all_installed = False
    
    return all_installed

def test_connection():
    """Test the actual database connection."""
    print("\n" + "=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    try:
        from FKGIS.backend.database.supabase_connection import connect_to_supabase, create_tables
        
        print("Attempting to connect to Supabase...")
        engine, SessionLocal = connect_to_supabase()
        print("[OK] Connection successful!")
        
        # Test a simple query
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"[OK] Database version: {version.split(',')[0]}")
            
            # Test if we can query
            result = conn.execute(text("SELECT current_database(), current_user"))
            db_info = result.fetchone()
            print(f"[OK] Connected to database: {db_info[0]}")
            print(f"[OK] Connected as user: {db_info[1]}")
        
        # Test table creation (won't fail if tables exist)
        print("\nTesting table creation...")
        try:
            create_tables()
            print("[OK] Tables verified/created successfully")
        except Exception as e:
            print(f"[WARN] Table creation warning: {e}")
        
        return True
        
    except ValueError as e:
        print(f"[FAIL] Configuration error: {e}")
        print("\nPlease check your .env file and ensure:")
        print("  1. SUPABASE_DATABASE_URL is set correctly, OR")
        print("  2. SUPABASE_URL and SUPABASE_DB_PASSWORD are set")
        import traceback
        print("\nFull error details:")
        traceback.print_exc()
        return False
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"[FAIL] Connection failed!")
        print(f"Error type: {error_type}")
        print(f"Error message: {error_msg}")
        
        # Check for common issues
        if "[PROJECT-REF]" in str(os.getenv('SUPABASE_DATABASE_URL', '')):
            print("\n[ISSUE] Your connection string still contains [PROJECT-REF] placeholder!")
            print("  Replace [PROJECT-REF] with your actual Supabase project reference")
        if "[YOUR-PASSWORD]" in str(os.getenv('SUPABASE_DATABASE_URL', '')):
            print("\n[ISSUE] Your connection string still contains [YOUR-PASSWORD] placeholder!")
            print("  Replace [YOUR-PASSWORD] with your actual database password")
        
        if "password authentication failed" in error_msg.lower() or "authentication failed" in error_msg.lower():
            print("\n[ISSUE] Authentication failed - Wrong password!")
            print("  Check your database password in Supabase Dashboard > Settings > Database")
        elif "could not translate host name" in error_msg.lower() or "name resolution" in error_msg.lower():
            print("\n[ISSUE] Cannot resolve hostname - Wrong project reference!")
            print("  Check your project reference in the connection string")
        elif "connection refused" in error_msg.lower() or "timeout" in error_msg.lower():
            print("\n[ISSUE] Connection refused or timed out!")
            print("  - Check if your Supabase project is active (not paused)")
            print("  - Check your network/firewall settings")
            print("  - Verify the host and port are correct")
        
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        return False

def test_basic_operations():
    """Test basic database operations."""
    print("\n" + "=" * 60)
    print("Testing Basic Database Operations")
    print("=" * 60)
    
    try:
        from FKGIS.backend.database.supabase_connection import get_db
        from FKGIS.backend.database.models.SQLmodels import Case
        from sqlalchemy.orm import Session
        
        db = next(get_db())
        
        # Test query
        case_count = db.query(Case).count()
        print(f"[OK] Can query database - Found {case_count} case(s) in database")
        
        # Test insert (rollback after)
        test_case = Case(
            case_id="TEST-CONNECTION",
            title="Test Connection Case",
            status="Test"
        )
        db.add(test_case)
        db.commit()
        print("[OK] Can insert data")
        
        # Clean up test data
        db.delete(test_case)
        db.commit()
        print("[OK] Can delete data")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"[FAIL] Operation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Supabase Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Environment variables
    results.append(("Environment Variables", test_environment_variables()))
    
    # Test 2: Package imports
    results.append(("Required Packages", test_imports()))
    
    # Test 3: Database connection
    if results[0][1] and results[1][1]:
        results.append(("Database Connection", test_connection()))
        
        # Test 4: Basic operations (only if connection works)
        if results[2][1]:
            results.append(("Basic Operations", test_basic_operations()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed! Supabase setup is complete.")
    else:
        print("[WARNING] Some tests failed. Please fix the issues above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

