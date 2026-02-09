import logging
import voyageai

from bson import ObjectId
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database as PyMongoDatabase
from pymongo.operations import SearchIndexModel

from services.env import MONGODB_URI, VOYAGE_API_KEY

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, ensure_indexes: bool = True):
        """Initialize database connection and clients"""
        logger.info("Connecting to MongoDB Client...")
        self._client: MongoClient = MongoClient(MONGODB_URI)
        self._db: PyMongoDatabase = self._client.get_default_database()
        self._voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY) if VOYAGE_API_KEY else None
        logger.info("MongoDB connected successfully")

        # Ensure vector search index exists
        if ensure_indexes:
            self._ensure_vector_search_index()

    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")

    def _ensure_vector_search_index(self):
        """
        Ensure vector search index exists on the tools collection.
        Creates it if it doesn't exist. Idempotent - safe to run multiple times.
        """
        try:
            collection = self._db.tools
            index_name = "search_index"

            # Check if index already exists
            existing_indexes = list(collection.list_search_indexes())
            if any(idx.get("name") == index_name for idx in existing_indexes):
                logger.info(f"Vector search index '{index_name}' already exists")
                return

            # Create the index
            logger.info(f"Creating vector search index '{index_name}'...")
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 1024,
                            "similarity": "dotProduct",
                            "quantization": "scalar",
                        }
                    ]
                },
                name=index_name,
                type="vectorSearch",
            )

            result = collection.create_search_index(model=search_index_model)
            logger.info(
                f"Vector search index '{result}' created. It may take a few minutes to build."
            )

        except Exception as e:
            # Log warning but don't fail startup
            logger.warning(
                f"Could not create vector search index (may already exist): {e}"
            )


    def generate_embedding(
        self, text: str, model: str = "voyage-4", input_type: str = "document"
    ) -> List[float]:
        """
        Generate embeddings for text using Voyage AI.
        If VOYAGE_API_KEY is not set, returns a zero vector so the app can run without embeddings.

        Args:
            text: Text to embed
            model: Embedding model to use (default: voyage-4, 1024 dimensions)
            input_type: Type of input - "query" for search queries, "document" for documents to search

        Returns:
            List of floats representing the embedding vector
        """
        if self._voyage_client is None:
            logger.debug("No Voyage API key; using zero vector for embedding")
            return [0.0] * 1024
        try:
            logger.info(f"Generating embedding for text: {text[:100]}...")
            result = self._voyage_client.embed(
                texts=[text], model=model, input_type=input_type
            )
            embedding = result.embeddings[0]
            logger.info(f"Generated embedding of dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.exception(f"Error generating embedding: {e}")
            logger.warning("Returning zero vector as fallback")
            return [0.0] * 1024


    def save_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """
        Save a conversation to MongoDB.

        Args:
            conversation_data: Dictionary containing conversation details

        Returns:
            Inserted conversation ID
        """
        conversations: Collection = self._db.conversations

        # Add timestamp if not present
        if "created_at" not in conversation_data:
            conversation_data["created_at"] = datetime.now(timezone.utc)

        result = conversations.insert_one(conversation_data)
        conversation_id = str(result.inserted_id)

        logger.info(f"Conversation saved with ID: {conversation_id}")
        return conversation_id


    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        conversations: Collection = self._db.conversations

        conversation = conversations.find_one({"_id": ObjectId(conversation_id)})

        if conversation:
            conversation["_id"] = str(conversation["_id"])

        return conversation


    def list_conversations(
        self, limit: int = 50, skip: int = 0
    ) -> List[Dict[str, Any]]:
        """List recent conversations"""
        conversations: Collection = self._db.conversations

        cursor = conversations.find().sort("created_at", -1).skip(skip).limit(limit)

        results = []
        for conv in cursor:
            conv["_id"] = str(conv["_id"])
            results.append(conv)

        return results


    def save_tool(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a generated tool to MongoDB with vector embeddings and enhanced metadata.

        Args:
            tool_data: Tool definition with name, description, parameters, code, and optional metadata

        Returns:
            Result dictionary with success status
        """
        try:
            logger.info(f"Starting save_tool for: {tool_data.get('name', 'unknown')}")

            tools: Collection = self._db.tools
            logger.info(f"Got database connection: {self._db.name}")

            # Add timestamp
            tool_data["created_at"] = datetime.now(timezone.utc)

            # TODO: Customize this
            tool_data.setdefault("status", "PROD-READY")
            tool_data.setdefault("category", "general")
            tool_data.setdefault("tags", [tool_data["name"]])
            tool_data.setdefault("verified", True)
            tool_data.setdefault("usage_count", 0)

            # Generate preview snippet if not provided
            if "preview_snippet" not in tool_data:
                params_str = ", ".join(
                    tool_data.get("parameters", {}).get("properties", {}).keys()
                )
                tool_data["preview_snippet"] = f"{tool_data['name']}({params_str})"

            # Generate embedding from tool name and description
            embedding_text = f"{tool_data['name']}: {tool_data['description']}"
            logger.info("Generating embedding...")
            embedding = self.generate_embedding(embedding_text)
            tool_data["embedding"] = embedding
            logger.info(f"Embedding generated, length: {len(embedding)}")

            # Remove _id field if present (can't update immutable _id field)
            update_data = {k: v for k, v in tool_data.items() if k != "_id"}

            # Update if exists, insert if new (upsert based on name)
            logger.info(f"Saving to MongoDB collection: {tools.name}")
            result = tools.update_one(
                {"name": update_data["name"]}, {"$set": update_data}, upsert=True
            )

            logger.info(
                f"Tool '{tool_data['name']}' saved to MongoDB. Upserted: {result.upserted_id is not None}, Modified: {result.modified_count}"
            )

            return {
                "success": True,
                "message": f"Tool '{tool_data['name']}' saved successfully",
                "has_code": "code" in tool_data and tool_data["code"] is not None,
                "upserted": result.upserted_id is not None,
            }
        except Exception as e:
            logger.exception(f"Error in save_tool: {e}")
            return {"success": False, "error": str(e)}


    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name"""
        tools: Collection = self._db.tools

        tool = tools.find_one({"name": name})

        if tool:
            tool["_id"] = str(tool["_id"])

        return tool


    def list_tools(self) -> List[Dict[str, Any]]:
        """List all tools"""
        tools: Collection = self._db.tools

        cursor = tools.find().sort("created_at", -1)

        results = []
        for tool in cursor:
            tool["_id"] = str(tool["_id"])
            results.append(tool)

        return results


    def delete_tool(self, name: str) -> bool:
        """Delete a tool by name"""
        tools: Collection = self._db.tools

        result = tools.delete_one({"name": name})
        deleted = result.deleted_count > 0

        if deleted:
            logger.info(f"Tool '{name}' deleted from MongoDB")

        return deleted


    def search_tools(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tools using vector similarity.
        Returns the top N most relevant tools based on the query.

        Args:
            query: Search query describing what the tool should do
            limit: Maximum number of tools to return (default: 10)

        Returns:
            List of tool definitions sorted by relevance
        """
        query_embedding = self.generate_embedding(query, input_type="query")

        # define pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "search_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 250,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "name": 1,
                    "description": 1,
                    "code": 1,
                    "parameters": 1,
                    "category": 1,
                    "status": 1,
                    "tags": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        # run pipeline
        cursor = self._db.tools.aggregate(pipeline)
        results = list(cursor)

        for tool in results:
            if "_id" in tool:
                tool["_id"] = str(tool["_id"])

        return results
