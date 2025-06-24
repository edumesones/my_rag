#!/usr/bin/env python3
"""
Quick test script for the RAG system to verify file upload and query functionality.
"""

import requests
import tempfile
import os
from pathlib import Path
import pandas as pd

def test_system():
    """Quick test of the RAG system."""
    base_url = "http://localhost:8000"
    api_base = f"{base_url}/api/rag"
    
    print("🚀 Quick RAG System Test")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{api_base}/healthcheck", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Create test Excel file
    print("\n2. Creating test Excel file...")
    try:
        data = {
            'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
            'Price': [1200, 25, 50, 300],
            'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics'],
            'Stock': [10, 50, 30, 15]
        }
        df = pd.DataFrame(data)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        df.to_excel(temp_file.name, index=False)
        temp_file.close()
        
        print(f"✅ Test file created: {temp_file.name}")
    except Exception as e:
        print(f"❌ Error creating test file: {e}")
        return False
    
    # Test 3: Upload file
    print("\n3. Testing file upload...")
    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test_products.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            response = requests.post(f"{api_base}/upload-file", files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File uploaded successfully: {result['filename']}")
            print(f"   Collection: {result['collection_name']}")
            collection_name = result['collection_name']
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    finally:
        # Clean up test file
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    # Test 4: Query the uploaded data
    print("\n4. Testing query...")
    try:
        query_payload = {
            "query": "What is the most expensive product?",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.1,
            "top_p": 1.0,
            "max_tokens": 512,
            "k": 3
        }
        
        response = requests.post(f"{api_base}/compiled-query", json=query_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful!")
            print(f"   Answer: {result['answer']}")
            print(f"   Context documents: {len(result['context'])}")
        else:
            print(f"❌ Query failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False
    
    # Test 5: Test retrieve-only
    print("\n5. Testing retrieve-only...")
    try:
        retrieve_payload = {
            "query": "electronics products",
            "k": 2
        }
        
        response = requests.post(f"{api_base}/retrieve-only", json=retrieve_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved {len(result['retrieved_docs'])} documents")
        else:
            print(f"❌ Retrieve-only failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Retrieve-only error: {e}")
        return False
    
    print("\n🎉 All tests passed! The RAG system is working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = test_system()
        if not success:
            print("\n❌ Some tests failed. Check the error messages above.")
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1) 