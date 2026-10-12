"""Serving layer: the dashboard and the assistant answering "will the PM limit be exceeded tomorrow in
city Y, and how sure is it?". Numbers come ONLY from gold tables through the SQL guardrails; a language
model (any OpenAI-compatible endpoint) only phrases results and answers from retrieved documents.
"""
