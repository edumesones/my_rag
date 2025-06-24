#!/usr/bin/env python3
"""
Test script for the RAG system file upload and query functionality.
"""

import requests
import json
import tempfile
import os
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/rag"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{API_BASE}/healthcheck")
    if response.status_code == 200:
        print("✅ Health check passed")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_list_models():
    """Test the list models endpoint."""
    print("🔍 Testing list models...")
    response = requests.get(f"{API_BASE}/list-models")
    if response.status_code == 200:
        models = response.json().get("models", [])
        print(f"✅ Found {len(models)} models: {models[:3]}...")
        return models
    else:
        print(f"❌ List models failed: {response.status_code}")
        return []

def create_test_excel_file():
    """Create a test Excel file."""
    import pandas as pd
    
    # Create test data
    data = {
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Age': [30, 25, 35],
        'Department': ['Engineering', 'Marketing', 'Sales'],
        'Salary': [75000, 65000, 80000]
    }
    
    df = pd.DataFrame(data)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    df.to_excel(temp_file.name, index=False)
    return temp_file.name

def test_file_upload(file_path):
    """Test file upload functionality."""
    print(f"🔍 Testing file upload: {Path(file_path).name}")
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        response = requests.post(f"{API_BASE}/upload-file", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ File uploaded successfully: {result['filename']}")
        print(f"   Collection: {result['collection_name']}")
        print(f"   File size: {result['file_size']} bytes")
        return result['collection_name']
    else:
        print(f"❌ File upload failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_query(collection_name):
    """Test querying uploaded documents."""
    print("🔍 Testing document query...")
    
    query_payload = {
        "query": "What is the average salary?",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_tokens": 512,
        "k": 3
    }
    
    response = requests.post(f"{API_BASE}/compiled-query", json=query_payload)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Query successful!")
        print(f"   Answer: {result['answer']}")
        print(f"   Context documents: {len(result['context'])}")
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_retrieve_only():
    """Test retrieve-only functionality."""
    print("🔍 Testing retrieve-only...")
    
    retrieve_payload = {
        "query": "salary information",
        "k": 2
    }
    
    response = requests.post(f"{API_BASE}/retrieve-only", json=retrieve_payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Retrieved {len(result['retrieved_docs'])} documents")
        for i, doc in enumerate(result['retrieved_docs']):
            print(f"   Doc {i+1}: {doc[:100]}...")
        return True
    else:
        print(f"❌ Retrieve-only failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting RAG system tests...\n")
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ Health check failed, stopping tests")
        return
    
    # Test 2: List models
    models = test_list_models()
    if not models:
        print("❌ Could not retrieve models, stopping tests")
        return
    
    # Test 3: Create and upload test file
    test_file = create_test_excel_file()
    try:
        collection_name = test_file_upload(test_file)
        if not collection_name:
            print("❌ File upload failed, stopping tests")
            return
        
        # Test 4: Query uploaded documents
        if not test_query(collection_name):
            print("❌ Query failed")
        
        # Test 5: Test retrieve-only
        test_retrieve_only()
        
    finally:
        # Clean up test file
        try:
            os.unlink(test_file)
        except:
            pass
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 