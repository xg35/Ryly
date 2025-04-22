class LLMManager:
    """Handles LLM interactions and prompt templating"""
    
    def __init__(self, llm, system_prompt):
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = []
    
    def generate(self, user_query, context, history_str, preferences):
        """Generate response using custom prompt template"""
        from langchain.schema import HumanMessage, SystemMessage
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
[Conversations]
{history_str}
[/Conversations]

[Customer Preferences]
{preferences}
[/Customer Preferences]

[Relevant Info]
{context}
[/Relevant Info]

[User says] 
{user_query}
[/User says]

Ryly's multi-message response (using ^MSG^ to separate):""")
        ]
        
        return self.llm(messages)
