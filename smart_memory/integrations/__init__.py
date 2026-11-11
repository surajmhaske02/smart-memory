"""Adapters for agent frameworks. Each module imports its framework lazily:

* ``smart_memory.integrations.langchain`` - retriever, chat history, context message (``langchain-core``)
* ``smart_memory.integrations.llamaindex`` - retriever (``llama-index-core``)
* ``smart_memory.integrations.crewai`` - remember / recall / context tools for CrewAI agents (``crewai``)
"""
