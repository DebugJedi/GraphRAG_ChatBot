from app.document_processor import DocumentProcessor
from app.knowledgegraph import knowledgeGraph
from app.visualizer import Visulaizer
from app.queryengine import QueryEngine
import streamlit as st
import numpy as np


class GraphRAG(DocumentProcessor):
    def __init__(self):
        """
            Initializes the GraphRAG system with components for document processing, knowledge graph construction,
            querying and visualization.

            Attributes: 
            - llm: An instance of large language model (LLM) for genrating responses.
            - embedding_model: An instance of embedding model for document embeddings.
            - documents_processor: An instance of DocumentProcessor class for processing documents.
            - knowledge_graph: An instance of the knowledgeGraph class for building and manging knowledge graph.
            - query_engine: An instance of the QueryEngine class for handling queries.
            - visulaizer : An instance of the Visuliazer class for visualizing the knowledge graph.
        """
        
        self.document_processor = DocumentProcessor()
        self.openai = DocumentProcessor().openai_model
        self.knowledge_graph = knowledgeGraph(openai_model=self.openai)
        self.query_engine = None
        self.visualizer = Visulaizer()
       

    def process_documents(self, documents):
        """
        Processes a list of documents by splitting them into chunks, embedding them, and building a knowledge graph.

        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - None
        """
        splits, vector_store, openai_model, documents = self.document_processor.process_documents(documents) 
        self.knowledge_graph.build_graph(splits)
        # st.write("Initiating Query Engine....")
        self.query_engine = QueryEngine(vector_store, self.knowledge_graph, self.openai, documents)

    def query(self, query: str):
        """
        Handles a query by retrieving relevant information from the knowledge graph and visulaizing the traversal path.

        Args:
        - query (str): The query to be answered.

        Returns:
        - str: The response to the query.
        """
        response, traversal_path, filtered_content = self.query_engine.query(query)

        # if traversal_path:
        #     self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        # else:
        #     print("No traversal path to visulaize.")

        return response

