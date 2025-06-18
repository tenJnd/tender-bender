import logging
from typing import List, Dict, Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# docker pull qdrant/qdrant
# docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
class VectorSearch:
    """Base vector search class using Qdrant and SentenceTransformers"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 index_definition: Dict[str, Any] = None):
        """
        Initialize the vector search with SentenceTransformer model and Qdrant client
        Args:
            model_name: Name of the SentenceTransformer model to use
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            index_definition: Dictionary of index definitions by entity type
        """
        logger.info(f"Initializing VectorSearch with model: {model_name}")
        # Initialize SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        # Initialize Qdrant client
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.index_definitions = index_definition if index_definition else {}
        # Track active collection versions
        self.active_versions = {}
        logger.info(f"VectorSearch initialized with vector size: {self.vector_size}")

    def _get_entity_type_from_collection_name(self, collection_name: str) -> Optional[str]:
        """
        Extract entity type from collection name (handles versioned collections)
        Args:
            collection_name: Name of the collection (e.g., 'tenders', 'tenders_v1.0', 'companies_v2.1')
        Returns:
            Entity type if recognized, None otherwise
        """
        # Check if collection name matches any known entity types exactly
        if collection_name in self.index_definitions:
            return collection_name

        # Check for versioned collections (entity_type_version format)
        for entity_type in self.index_definitions.keys():
            if collection_name.startswith(f"{entity_type}_"):
                return entity_type

        return None

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """
        Ensure that a collection exists, create it if it doesn't
        Args:
            collection_name: Name of the collection to ensure exists
        """
        try:
            self.qdrant.get_collection(collection_name)
        except Exception:
            logger.info(f"Collection '{collection_name}' doesn't exist, creating it")
            self.create_collection(collection_name)

    @staticmethod
    def _create_point(item: Dict[str, Any], vector: np.ndarray) -> PointStruct:
        """
        Create a PointStruct from item data and vector
        Args:
            item: Item with 'id' and optional 'payload'
            vector: Vector representation
        Returns:
            PointStruct ready for Qdrant
        """
        return PointStruct(
            id=item['id'],
            vector=vector.tolist(),
            payload=item.get('payload', {})
        )

    def create_collection(self, collection_name: str, vector_size: Optional[int] = None,
                          distance: Distance = Distance.COSINE, force_recreate: bool = False) -> bool:
        """
        Create a new Qdrant collection with automatic index creation based on entity type
        Args:
            collection_name: Name of the collection to create
            vector_size: Size of vectors (uses model's embedding dimension if not provided)
            distance: Distance metric to use
            force_recreate: Whether to recreate if collection already exists
        Returns:
            True if collection was created or already exists, False otherwise
        """
        try:
            # Check if collection already exists
            try:
                self.qdrant.get_collection(collection_name)
                if not force_recreate:
                    logger.info(f"Collection '{collection_name}' already exists")
                    return True
                else:
                    logger.info(f"Deleting existing collection '{collection_name}' for recreation")
                    self.qdrant.delete_collection(collection_name)
            except Exception:
                # Collection doesn't exist, which is fine
                pass

            # Use provided vector size or default to model's embedding dimension
            if vector_size is None:
                vector_size = self.vector_size

            # Create the collection
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")

            # Check if we have index definitions for this collection and create indexes
            entity_type = self._get_entity_type_from_collection_name(collection_name)
            if entity_type and entity_type in self.index_definitions:
                pass # TODO fix this
                # self._create_collection_indexes(collection_name, entity_type)

            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False

    def _create_collection_indexes(self, collection_name: str, entity_type: str) -> None:
        """
        Create payload indexes for a collection based on entity type
        Args:
            collection_name: Name of the collection
            entity_type: Type of entity ('tenders' or 'companies')
        """
        indexes = self.index_definitions[entity_type]
        for field_name, field_type in indexes.indexes:
            try:
                self.create_payload_index(collection_name, field_name, field_type)
                logger.info(f"Created index on {field_name} ({field_type}) for {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to create index on {field_name}: {e}")

    def create_collection_with_indexes(self, collection_name: str, entity_type: str,
                                       force_recreate: bool = False) -> bool:
        """
        Create a collection with appropriate payload indexes (legacy method - use create_collection instead)
        Args:
            collection_name: Name of the collection to create
            entity_type: Type of entity ('tenders' or 'companies')
            force_recreate: Whether to recreate if collection already exists
        Returns:
            True if collection was created successfully, False otherwise
        """
        logger.warning(
            "create_collection_with_indexes is deprecated. "
            "Use create_collection instead - indexes are created automatically."
        )
        return self.create_collection(collection_name, force_recreate=force_recreate)

    def create_payload_index(self, collection_name: str, field_name: str,
                             field_type: str = "keyword") -> bool:
        """
        Create an index on a payload field for faster filtering
        Args:
            collection_name: Name of the collection
            field_name: Name of the payload field to index
            field_type: Type of the field (keyword, integer, float, geo, text)
        Returns:
            True if index was created successfully, False otherwise
        """
        try:
            self.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            logger.info(f"Created index on field '{field_name}' in collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create index on field '{field_name}': {e}")
            return False

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into a vector
        Args:
            text: Text to encode
        Returns:
            Vector representation of the text
        """
        try:
            vector = self.model.encode(text)
            return vector
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into vectors
        Args:
            texts: List of texts to encode
        Returns:
            List of vector representations
        """
        try:
            vectors = self.model.encode(texts)
            return [vector for vector in vectors]
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise

    def add_items(self, collection_name: str, items: List[Dict[str, Any]],
                  batch_size: int = 100) -> List[str]:
        """
        Add preprocessed items to a collection
        Args:
            collection_name: Name of the collection
            items: List of preprocessed items, each with 'id', 'text', and optional 'payload'
            batch_size: Number of items to process in each batch
        Returns:
            List of IDs of added items
        """
        try:
            # Ensure collection exists
            self._ensure_collection_exists(collection_name)
            added_ids = []
            # Process items in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                points = []
                # Prepare texts for batch encoding
                texts = [item['text'] for item in batch]
                vectors = self.encode_batch(texts)
                # Create points
                for item, vector in zip(batch, vectors):
                    points.append(self._create_point(item, vector))
                # Upload batch to Qdrant
                self.qdrant.upsert(
                    collection_name=collection_name,
                    points=points
                )
                batch_ids = [item['id'] for item in batch]
                added_ids.extend(batch_ids)
                logger.info(f"Added batch of {len(batch)} items to '{collection_name}'")
            logger.info(f"Successfully added {len(added_ids)} items to collection '{collection_name}'")
            return added_ids
        except Exception as e:
            logger.error(f"Failed to add items to collection '{collection_name}': {e}")
            raise

    def add_item(self, collection_name: str, item: Dict[str, Any]) -> str:
        """
        Add a single preprocessed item to a collection
        Args:
            collection_name: Name of the collection
            item: Preprocessed item with 'id', 'text', and optional 'payload'
        Returns:
            ID of the added item
        """
        return self.add_items(collection_name, [item])[0]

    def upsert_item(self, collection_name: str, item: Dict[str, Any]) -> str:
        """
        Upsert (insert or update) a single preprocessed item
        Args:
            collection_name: Name of the collection
            item: Preprocessed item with 'id', 'text', and optional 'payload'
        Returns:
            ID of the upserted item
        """
        try:
            # Ensure collection exists
            self._ensure_collection_exists(collection_name)
            # Encode the text and create point
            vector = self.encode_text(item['text'])
            point = self._create_point(item, vector)
            # Upsert to Qdrant
            self.qdrant.upsert(
                collection_name=collection_name,
                points=[point]
            )
            logger.info(f"Upserted item '{item['id']}' to collection '{collection_name}'")
            return item['id']
        except Exception as e:
            logger.error(f"Failed to upsert item to collection '{collection_name}': {e}")
            raise

    def query_points(self, collection_name: str, query_text: str, top_k: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar items in a collection using the current Qdrant API
        Args:
            collection_name: Name of the collection to search
            query_text: Text to search for
            top_k: Number of results to return
            filters: Optional filters to apply
        Returns:
            List of search results with id, score, and payload
        """
        try:
            # Encode the query text
            query_vector = self.encode_text(query_text)
            # Build filter if provided
            query_filter = self._build_filter(filters) if filters else None
            # Perform search using the current API method
            search_results = self.qdrant.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                query_filter=query_filter,
                limit=top_k
            )
            # Format results
            results = []
            for result in search_results.points:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })
            logger.info(f"Found {len(results)} results in collection '{collection_name}'")
            return results
        except Exception as e:
            logger.error(f"Failed to query collection '{collection_name}': {e}")
            raise

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> Filter:
        """
        Helper method to build a Qdrant filter object from a dictionary
        Args:
            filters: Dictionary of field-value pairs for filtering
        Returns:
            Qdrant Filter object
        """
        if not filters:
            return None
        must_conditions = []
        for field, value in filters.items():
            if isinstance(value, list):
                # Match any value in the list - use MatchAny for arrays
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchAny(any=value)
                    )
                )
            elif isinstance(value, str):
                # Exact string match
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            elif isinstance(value, (int, float, bool)):
                # Exact value match
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
        return Filter(must=must_conditions)

    def get_collection_name(self, entity_type: str, version: Optional[str] = None) -> str:
        """
        Generate collection name with optional versioning
        Args:
            entity_type: Type of entity (e.g., 'tenders', 'companies')
            version: Optional version string
        Returns:
            Collection name
        """
        if version:
            return f"{entity_type}_{version}"
        return entity_type

    def update_payload(self, collection_name: str, point_id: str, payload: Dict[str, Any]) -> bool:
        """
        Update the payload of a specific point
        Args:
            collection_name: Name of the collection
            point_id: ID of the point to update
            payload: New payload data
        Returns:
            True if update was successful, False otherwise
        """
        try:
            self.qdrant.set_payload(
                collection_name=collection_name,
                points=[point_id],
                payload=payload
            )
            logger.info(f"Updated payload for point '{point_id}' in collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to update payload for point '{point_id}': {e}")
            return False


class EnhancedVectorSearch(VectorSearch):
    """Enhanced vector search with caching capabilities"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 index_definition: Dict[str, Any] = None):
        """
        Initialize the enhanced vector search
        """
        super().__init__(model_name, qdrant_host, qdrant_port, index_definition)
        self.cache = {}
        logger.info("EnhancedVectorSearch initialized with caching")

    def add_item(self, collection_name: str, item: Dict[str, Any]) -> str:
        """
        Add a single preprocessed item with caching
        Args:
            collection_name: Name of the collection
            item: Preprocessed item with 'id', 'text', and optional 'payload'
        Returns:
            ID of the added item
        """
        # Add to vector database
        item_id = super().add_item(collection_name, item)
        # Cache the item
        if collection_name not in self.cache:
            self.cache[collection_name] = {}
        self.cache[collection_name][item_id] = item
        return item_id

    def add_items(self, collection_name: str, items: List[Dict[str, Any]],
                  batch_size: int = 100) -> List[str]:
        """
        Add preprocessed items with caching
        Args:
            collection_name: Name of the collection
            items: List of preprocessed items, each with 'id', 'text', and optional 'payload'
            batch_size: Number of items to process in each batch
        Returns:
            List of IDs of added items
        """
        # Add to vector database
        added_ids = super().add_items(collection_name, items, batch_size)
        # Cache the items
        if collection_name not in self.cache:
            self.cache[collection_name] = {}
        for item in items:
            self.cache[collection_name][item['id']] = item
        return added_ids

    def upsert_item(self, collection_name: str, item: Dict[str, Any]) -> str:
        """
        Upsert (insert or update) a single preprocessed item with caching
        Args:
            collection_name: Name of the collection
            item: Preprocessed item with 'id', 'text', and optional 'payload'
        Returns:
            ID of the upserted item
        """
        # Upsert to vector database
        item_id = super().upsert_item(collection_name, item)
        # Update cache
        if collection_name not in self.cache:
            self.cache[collection_name] = {}
        self.cache[collection_name][item_id] = item
        return item_id

    def get_cached_item(self, collection_name: str, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an item from cache
        Args:
            collection_name: Name of the collection
            item_id: ID of the item
        Returns:
            Cached item or None if not found
        """
        return self.cache.get(collection_name, {}).get(item_id)

    def clear_cache(self, collection_name: Optional[str] = None) -> None:
        """
        Clear cache for a specific collection or all collections
        Args:
            collection_name: Name of the collection to clear, or None to clear all
        """
        if collection_name:
            self.cache.pop(collection_name, None)
            logger.info(f"Cleared cache for collection '{collection_name}'")
        else:
            self.cache.clear()
            logger.info("Cleared all cache")
