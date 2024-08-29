from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Dict

class ExplanationAgent:
    def __init__(self, openai_api_key: str):
        self.llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.account_db = None

    def create_account_database(self, accounts_data: List[Dict]):
        texts = [str(account) for account in accounts_data]
        self.account_db = FAISS.from_texts(texts, self.embeddings)

    def generate_explanation(self, model_name: str, prediction: float, top_features: List[str], account_id: str):
        account_info = self.account_db.similarity_search(f"Account ID: {account_id}", k=1)[0].page_content

        template = """
        As an AI assistant for a CRM system, your task is to explain the {model_name} prediction for an account.

        Account Information:
        {account_info}

        Prediction: {prediction}

        Top Contributing Factors:
        {top_features}

        Please provide a 2-3 paragraph summary explaining the prediction, focusing on how the top contributing factors relate to the account's data and usage patterns. Be specific and use concrete examples from the account information when possible.

        Summary:
        """

        prompt = PromptTemplate(
            input_variables=["model_name", "account_info", "prediction", "top_features"],
            template=template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        return chain.run(model_name=model_name, account_info=account_info, prediction=prediction, top_features=top_features)