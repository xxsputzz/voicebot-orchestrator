"""
Conversation Memory Manager
Handles conversation context, prevents repeated introductions, tracks conversation state
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import uuid

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation context and memory"""
    
    def __init__(self, db_path: Optional[str] = None):
        # Default database path
        if db_path is None:
            db_path = Path(__file__).parent.parent / "conversations.db"
        
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize conversation database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    customer_phone TEXT,
                    call_type TEXT,
                    conversation_state TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_phone 
                ON conversations (customer_phone, created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages (conversation_id, timestamp)
            """)
    
    def start_conversation(self, customer_phone: str = None, call_type: str = "general") -> str:
        """Start a new conversation and return conversation_id"""
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO conversations 
                (conversation_id, customer_phone, call_type, conversation_state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (conversation_id, customer_phone, call_type, "greeting", now, now))
        
        logger.info(f"Started conversation {conversation_id} (call_type: {call_type})")
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to the conversation"""
        now = datetime.now()
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages (conversation_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (conversation_id, role, content, now, metadata_json))
            
            # Update conversation timestamp
            conn.execute("""
                UPDATE conversations 
                SET updated_at = ? 
                WHERE conversation_id = ?
            """, (now, conversation_id))
    
    def get_conversation_history(self, conversation_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (conversation_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                message = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"]
                }
                if row["metadata"]:
                    message["metadata"] = json.loads(row["metadata"])
                messages.append(message)
            
            return list(reversed(messages))  # Return in chronological order
    
    def get_conversation_context(self, conversation_id: str) -> Dict:
        """Get conversation context including state and recent history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get conversation info
            cursor = conn.execute("""
                SELECT * FROM conversations 
                WHERE conversation_id = ?
            """, (conversation_id,))
            
            conv_info = cursor.fetchone()
            if not conv_info:
                return None
            
            # Get message count
            cursor = conn.execute("""
                SELECT COUNT(*) as message_count
                FROM messages
                WHERE conversation_id = ?
            """, (conversation_id,))
            
            message_count = cursor.fetchone()["message_count"]
            
            # Get recent history
            history = self.get_conversation_history(conversation_id, 10)
            
            return {
                "conversation_id": conversation_id,
                "customer_phone": conv_info["customer_phone"],
                "call_type": conv_info["call_type"],
                "conversation_state": conv_info["conversation_state"],
                "created_at": conv_info["created_at"],
                "updated_at": conv_info["updated_at"],
                "message_count": message_count,
                "is_first_interaction": message_count == 0,
                "recent_history": history
            }
    
    def update_conversation_state(self, conversation_id: str, state: str):
        """Update conversation state (greeting, qualifying, objection_handling, transfer, etc.)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE conversations 
                SET conversation_state = ?, updated_at = ?
                WHERE conversation_id = ?
            """, (state, datetime.now(), conversation_id))
    
    def find_active_conversation(self, customer_phone: str, hours_ago: int = 2) -> Optional[str]:
        """Find active conversation for customer within time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours_ago)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT conversation_id
                FROM conversations
                WHERE customer_phone = ? 
                AND is_active = 1
                AND updated_at > ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (customer_phone, cutoff_time))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def close_conversation(self, conversation_id: str):
        """Mark conversation as inactive"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE conversations 
                SET is_active = 0, updated_at = ?
                WHERE conversation_id = ?
            """, (datetime.now(), conversation_id))
    
    def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up conversations older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete old messages first (foreign key constraint)
            conn.execute("""
                DELETE FROM messages 
                WHERE conversation_id IN (
                    SELECT conversation_id FROM conversations 
                    WHERE updated_at < ?
                )
            """, (cutoff_time,))
            
            # Delete old conversations
            cursor = conn.execute("""
                DELETE FROM conversations 
                WHERE updated_at < ?
            """, (cutoff_time,))
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleaned up {deleted_count} old conversations")
            return deleted_count

# Global conversation manager instance
conversation_manager = ConversationManager()
