#!/usr/bin/env python3
"""
Conversation Manager Test Suite
Comprehensive testing of the SQLite-based conversation management system.
"""
import sys
import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime

# Add aws_microservices to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aws_microservices"))

try:
    from conversation_manager import ConversationManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

class ConversationManagerTester:
    """Test suite for ConversationManager functionality"""
    
    def __init__(self):
        """Initialize the tester"""
        self.test_db_path = "test_conversations.db"
        self.cleanup_test_db()
        
        print("🗄️  Conversation Manager Test Suite")
        print("=" * 50)
    
    def cleanup_test_db(self):
        """Clean up test database before and after testing"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
            print(f"🧹 Cleaned up previous test database: {self.test_db_path}")
    
    def test_database_initialization(self):
        """Test database creation and table setup"""
        print(f"\n🔧 Testing Database Initialization")
        print("-" * 40)
        
        # Create conversation manager with test database
        manager = ConversationManager(db_path=self.test_db_path)
        
        # Check if database file was created
        if os.path.exists(self.test_db_path):
            print("  ✅ Database file created successfully")
        else:
            print("  ❌ Database file not created")
            return False
        
        # Check if tables exist
        try:
            with sqlite3.connect(self.test_db_path) as conn:
                cursor = conn.cursor()
                
                # Check conversations table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'")
                if cursor.fetchone():
                    print("  ✅ Conversations table created")
                else:
                    print("  ❌ Conversations table missing")
                    return False
                
                # Check messages table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
                if cursor.fetchone():
                    print("  ✅ Messages table created")
                else:
                    print("  ❌ Messages table missing")
                    return False
                
                print("  ✅ Database initialization test passed")
                return True
                
        except Exception as e:
            print(f"  ❌ Database initialization test failed: {e}")
            return False
    
    def test_conversation_creation(self):
        """Test conversation creation functionality"""
        print(f"\n💬 Testing Conversation Creation")
        print("-" * 40)
        
        manager = ConversationManager(db_path=self.test_db_path)
        
        test_cases = [
            ("555-0001", "inbound", "Test inbound call"),
            ("555-0002", "outbound", "Test outbound call"),
            ("555-0003", None, "Test general call"),
            ("555-0001", "inbound", "Test duplicate customer")
        ]
        
        conversation_ids = []
        
        for phone, call_type, description in test_cases:
            try:
                conversation_id = manager.start_conversation(phone, call_type)
                conversation_ids.append(conversation_id)
                
                if conversation_id:
                    print(f"  ✅ {description}: {conversation_id[:8]}...")
                else:
                    print(f"  ❌ {description}: No conversation ID returned")
                    return False
                    
            except Exception as e:
                print(f"  ❌ {description}: Error - {e}")
                return False
        
        # Verify conversations in database
        try:
            with sqlite3.connect(self.test_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conversations")
                count = cursor.fetchone()[0]
                
                print(f"  📊 Created {count} conversations in database")
                if count == len(test_cases):
                    print("  ✅ Conversation creation test passed")
                    return conversation_ids
                else:
                    print(f"  ❌ Expected {len(test_cases)} conversations, found {count}")
                    return False
                    
        except Exception as e:
            print(f"  ❌ Database verification failed: {e}")
            return False
    
    def test_message_management(self, conversation_ids):
        """Test message adding and retrieval"""
        print(f"\n📝 Testing Message Management")
        print("-" * 40)
        
        if not conversation_ids:
            print("  ❌ No conversation IDs available for testing")
            return False
        
        manager = ConversationManager(db_path=self.test_db_path)
        test_conversation = conversation_ids[0]
        
        # Test message scenarios
        messages = [
            ("user", "Hello, I need help with my debt."),
            ("assistant", "Hi, I'm Alex with Finally Payoff Debt. How can I help you?"),
            ("user", "I have $25,000 in credit card debt."),
            ("assistant", "I can help you consolidate that debt with a personal loan."),
            ("user", "What are the rates?"),
            ("assistant", "Our rates start at 5.99% APR for qualified borrowers.")
        ]
        
        # Add messages
        for role, content in messages:
            try:
                manager.add_message(test_conversation, role, content)
                print(f"  ✅ Added {role} message: {content[:30]}...")
            except Exception as e:
                print(f"  ❌ Failed to add {role} message: {e}")
                return False
        
        # Test message retrieval
        try:
            history = manager.get_conversation_history(test_conversation)
            
            if len(history) == len(messages):
                print(f"  ✅ Retrieved {len(history)} messages from history")
                
                # Verify message order and content
                for i, (role, content) in enumerate(messages):
                    if (history[i]["role"] == role and 
                        history[i]["content"] == content):
                        continue
                    else:
                        print(f"  ❌ Message {i+1} doesn't match expected content")
                        return False
                
                print("  ✅ Message content verification passed")
                
            else:
                print(f"  ❌ Expected {len(messages)} messages, got {len(history)}")
                return False
                
        except Exception as e:
            print(f"  ❌ Message retrieval failed: {e}")
            return False
        
        # Test limited history retrieval
        try:
            limited_history = manager.get_conversation_history(test_conversation, limit=3)
            if len(limited_history) == 3:
                print("  ✅ Limited history retrieval (limit=3) works")
            else:
                print(f"  ❌ Limited history returned {len(limited_history)} instead of 3")
                return False
        except Exception as e:
            print(f"  ❌ Limited history retrieval failed: {e}")
            return False
        
        print("  ✅ Message management test passed")
        return True
    
    def test_conversation_context(self, conversation_ids):
        """Test conversation context retrieval"""
        print(f"\n🧠 Testing Conversation Context")
        print("-" * 40)
        
        if not conversation_ids:
            print("  ❌ No conversation IDs available for testing")
            return False
        
        manager = ConversationManager(db_path=self.test_db_path)
        test_conversation = conversation_ids[0]
        
        try:
            context = manager.get_conversation_context(test_conversation)
            
            if context:
                print("  ✅ Conversation context retrieved successfully")
                
                # Check expected fields
                expected_fields = ["conversation_id", "customer_phone", "call_type", 
                                 "conversation_state", "message_count", "recent_history"]
                
                for field in expected_fields:
                    if field in context:
                        print(f"    ✅ {field}: {context[field]}")
                    else:
                        print(f"    ❌ Missing field: {field}")
                        return False
                
                # Verify message count matches what we added
                if context["message_count"] > 0:
                    print(f"  ✅ Message count verification: {context['message_count']} messages")
                else:
                    print("  ❌ No messages found in context")
                    return False
                
                print("  ✅ Conversation context test passed")
                return True
                
            else:
                print("  ❌ No conversation context returned")
                return False
                
        except Exception as e:
            print(f"  ❌ Conversation context test failed: {e}")
            return False
    
    def test_conversation_search(self):
        """Test conversation search and lookup functionality"""
        print(f"\n🔍 Testing Conversation Search")
        print("-" * 40)
        
        manager = ConversationManager(db_path=self.test_db_path)
        
        # Test finding conversations by customer phone
        test_phone = "555-0001"
        
        try:
            # Get all conversations (this tests the basic database query)
            with sqlite3.connect(self.test_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT conversation_id, customer_phone, call_type 
                    FROM conversations 
                    WHERE customer_phone = ?
                    ORDER BY created_at DESC
                """, (test_phone,))
                
                conversations = cursor.fetchall()
                
                if conversations:
                    print(f"  ✅ Found {len(conversations)} conversations for {test_phone}")
                    for conv_id, phone, call_type in conversations:
                        print(f"    📞 {conv_id[:8]}... | {phone} | {call_type}")
                else:
                    print(f"  ⚠️  No conversations found for {test_phone}")
                
                # Test database statistics
                cursor.execute("SELECT COUNT(*) FROM conversations")
                total_conversations = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM messages")
                total_messages = cursor.fetchone()[0]
                
                print(f"  📊 Database statistics:")
                print(f"    📝 Total conversations: {total_conversations}")
                print(f"    💬 Total messages: {total_messages}")
                
                print("  ✅ Conversation search test passed")
                return True
                
        except Exception as e:
            print(f"  ❌ Conversation search test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all conversation manager tests"""
        print("🚀 Starting Complete Conversation Manager Test Suite")
        print("=" * 60)
        
        test_results = []
        
        # Test 1: Database initialization
        result1 = self.test_database_initialization()
        test_results.append(("Database Initialization", result1))
        
        # Test 2: Conversation creation
        conversation_ids = self.test_conversation_creation()
        result2 = bool(conversation_ids)
        test_results.append(("Conversation Creation", result2))
        
        if not result2:
            conversation_ids = []
        
        # Test 3: Message management
        result3 = self.test_message_management(conversation_ids)
        test_results.append(("Message Management", result3))
        
        # Test 4: Conversation context
        result4 = self.test_conversation_context(conversation_ids)
        test_results.append(("Conversation Context", result4))
        
        # Test 5: Conversation search
        result5 = self.test_conversation_search()
        test_results.append(("Conversation Search", result5))
        
        # Print test summary
        print(f"\n📊 Test Results Summary")
        print("=" * 60)
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name:<25} {status}")
            if result:
                passed += 1
        
        total_tests = len(test_results)
        print(f"\n🎯 Overall Result: {passed}/{total_tests} tests passed")
        
        if passed == total_tests:
            print("🎉 All conversation manager tests passed!")
        else:
            print(f"⚠️  {total_tests - passed} tests failed")
        
        # Cleanup
        print(f"\n🧹 Cleaning up test database...")
        self.cleanup_test_db()
        
        return passed == total_tests

def main():
    """Main test execution"""
    try:
        tester = ConversationManagerTester()
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
