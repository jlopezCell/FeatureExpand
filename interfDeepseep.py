import os
import pandas as pd
import sympy as sp
import gradio as gr
import logging
import requests
from crewai import Agent, Task, Crew
from langchain.embeddings.base import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Implementación personalizada de DeepSeekEmbeddings
class DeepSeekEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/embeddings"  # URL de la API de DeepSeek

    def embed_documents(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "inputs": texts
        }
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["data"]
        else:
            raise Exception(f"Error en la API de DeepSeek: {response.status_code} - {response.text}")

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# 2. Implementación personalizada de ChatDeepSeek
class ChatDeepSeek(BaseChatModel):
    def __init__(self, model_name, api_key):
        super().__init__()  # Ensure the parent class is initialized
        self._model_name = model_name  # Use a private attribute to avoid conflicts
        self._api_key = api_key  # Use a private attribute to avoid conflicts
        self._base_url = "https://api.deepseek.com/v1"  # Use a private attribute for base_url
        
        ##self._base_url = "https://api.deepseek.com/v1/chat/completions"  # Use a private attribute for base_url

    def _generate(self, messages, stop=None):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self._model_name,
            "messages": messages
        }
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error en la API de DeepSeek: {response.status_code} - {response.text}")

    def _llm_type(self):
        return "deepseek"    

# 3. Sistema de Gestión Documental
class DocumentAnalyzer:
    def __init__(self, api_key):
        self.vector_store = None
        self.embeddings = DeepSeekEmbeddings(api_key)  # Usamos DeepSeekEmbeddings
    
    def upload_document(self, file_path):
        if self.vector_store:
            return "✅ Documentos ya cargados."
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            self.vector_store = FAISS.from_documents(pages, self.embeddings)
            return "✅ Documentos analizados exitosamente!"
        except FileNotFoundError:
            return "❌ Error: Archivo no encontrado."
        except Exception as e:
            return f"❌ Error inesperado: {str(e)}"
    
    def query_documents(self, question):
        if not self.vector_store:
            return "Primero cargue documentos técnicos"
        docs = self.vector_store.similarity_search(question, k=3)
        return "\n\n".join([f"📄 Página {d.metadata['page']}:\n{d.page_content}" for d in docs])

# 4. Sistema de Agentes Especializados
class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description
        
class MLAgents:
    def __init__(self, document_analyzer, api_key):
        self.document_analyzer = document_analyzer
        self.llm = ChatDeepSeek("deepseek-chat", api_key=api_key)  # Usamos ChatDeepSeek
        self.explanation_cache = {}
        
        # Define tools
        boolean_feature_tool = Tool(
            name="boolean_feature_tool",
            func=self.boolean_feature_tool,
            description="Generates and simplifies optimal logical rules."
        )
        
        document_query_tool = Tool(
            name="document_query_tool",
            func=self.document_query_tool,
            description="Queries technical documents for validation."
        )
        
        model_optimization_tool = Tool(
            name="model_optimization_tool",
            func=self.model_optimization_tool,
            description="Optimizes linear models with regularization."
        )
        
        explanation_generation_tool = Tool(
            name="explanation_generation_tool",
            func=self.explanation_generation_tool,
            description="Generates contextual explanations for rules."
        )
        
        self.agents = {
            'feature_engineer': Agent(
                role='Ingeniero de Características Booleanas',
                goal='Generar y simplificar reglas lógicas óptimas',
                backstory="Experto en lógica booleana y simplificación de expresiones.",
                tools=[boolean_feature_tool],
                verbose=True
            ),
            'domain_validator': Agent(
                role='Validador de Dominio',
                goal='Validar reglas con documentos técnicos',
                backstory="Especialista en validación de reglas con documentos técnicos.",
                tools=[document_query_tool],
                verbose=True
            ),
            'model_optimizer': Agent(
                role='Optimizador de Modelo',
                goal='Optimizar modelo lineal con regularización',
                backstory="Experto en optimización de modelos lineales.",
                tools=[model_optimization_tool],
                verbose=True
            ),
            'explanation_expert': Agent(
                role='Experto en Explicaciones',
                goal='Generar explicaciones contextuales',
                backstory="Especialista en generar explicaciones claras y contextuales.",
                tools=[explanation_generation_tool],
                verbose=True
            )
        }
    
    def boolean_feature_tool(self, data):
        a, b = sp.symbols('A B')
        expressions = [
            a & b, a | b, ~a, ~b,
            a & ~b, ~a & b, a ^ b
        ]
        simplified = list({sp.simplify(expr) for expr in expressions})
        return [str(expr) for expr in simplified]
    
    def document_query_tool(self, query):
        return self.document_analyzer.query_documents(query)
    
    def model_optimization_tool(self, features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
        model = Ridge(alpha=1.0)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        return model, mse, scores.mean()
    
    def explanation_generation_tool(self, rule):
        if rule in self.explanation_cache:
            return self.explanation_cache[rule]
        
        context = self.document_analyzer.query_documents(rule)
        prompt = f"""
        Contexto del dominio:
        {context}
        
        Explica esta regla booleana para no expertos: {rule}
        """
        explanation = self.llm._generate([{"role": "user", "content": prompt}])
        self.explanation_cache[rule] = explanation
        return explanation

# 5. Flujo de Trabajo Principal
class MLWorkflow:
    def __init__(self, api_key):
        self.document_analyzer = DocumentAnalyzer(api_key)
        self.agents = MLAgents(self.document_analyzer, api_key)
        self.current_project = None
    
    def create_task(self, description, agent_name, expected_output, context=None):
        return Task(
            description=description,
            agent=self.agents.agents[agent_name],
            expected_output=expected_output,
            context=context
        )
    
    def run_pipeline(self, data, documents=None):
        try:
            logger.info("Iniciando pipeline...")
            if documents:
                logger.info("Cargando documentos...")
                doc_status = self.document_analyzer.upload_document(documents.name)
                if "❌" in doc_status:
                    logger.error(doc_status)
                    return {}, doc_status  # Return empty dict for tabbedinterface and error message for textbox
                
            logger.info("Generando características booleanas...")
            feature_task = self.create_task(
                "Generar características booleanas para los datos",
                'feature_engineer',
                "Lista de reglas booleanas simplificadas"
            )
            
            logger.info("Validando características...")
            validation_task = self.create_task(
                "Validar reglas con conocimiento del dominio",
                'domain_validator',
                "Reglas validadas con documentos técnicos",
                [feature_task]
            )
            
            logger.info("Optimizando modelo...")
            optimization_task = self.create_task(
                "Entrenar y optimizar modelo lineal",
                'model_optimizer',
                "Modelo entrenado con métricas de rendimiento",
                [validation_task]
            )
            
            logger.info("Generando explicaciones...")
            explanation_task = self.create_task(
                "Crear explicaciones contextuales para cada regla",
                'explanation_expert',
                "Reporte explicativo completo",
                [optimization_task]
            )
            
            crew = Crew(
                agents=list(self.agents.agents.values()),
                tasks=[feature_task, validation_task, optimization_task, explanation_task],
                verbose=2
            )
            
            result = crew.kickoff()
            return result, "✅ Pipeline ejecutado exitosamente!"
        
        except Exception as e:
            logger.error(f"Error en el pipeline: {str(e)}")
            return {}, f"❌ Error en el pipeline: {str(e)}"  # Return empty dict for tabbedinterface and error message for textbox


# 6. Interfaz de Usuario
def create_gradio_interface(api_key):
    workflow = MLWorkflow(api_key)
    
    with gr.Blocks(title="ML Explicable con Agentes Especializados", theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown("# 🚀 ML Explicable con Agentes Especializados")
        gr.Markdown("Sistema multi-agente para generar modelos lineales interpretables con explicaciones contextuales")
        
        # Sección de Entrada
        with gr.Row():
            with gr.Column(scale=1):
                data_input = gr.File(label="1. Subir Dataset (CSV)", file_types=[".csv"])
                doc_input = gr.File(label="2. Subir Documentos Técnicos (PDF)", file_types=[".pdf"])
                run_btn = gr.Button("▶️ Ejecutar Pipeline Completo", variant="primary")
                status = gr.Textbox(label="Estado", interactive=False)
            
            with gr.Column(scale=2):
                output_panel = gr.TabbedInterface(
                    [
                        gr.Interface(
                            lambda: None,
                            inputs=None,
                            outputs=gr.JSON(label="Resultados Técnicos"),
                            title="Métricas"
                        ),
                        gr.Interface(
                            lambda: None,
                            inputs=None,
                            outputs=gr.Textbox(label="Explicaciones", lines=10),
                            title="Explicaciones"
                        ),
                        gr.Interface(
                            lambda: None,
                            inputs=None,
                            outputs=gr.DataFrame(label="Reglas Generadas"),
                            title="Reglas"
                        )
                    ],
                    tab_names=["Métricas", "Explicaciones", "Reglas"]
                )
        
        # Sección de Chat
        def chat_response(msg, history, data_input, doc_input):
            # You can use data_input and doc_input if needed in the response
            return workflow.agents.explanation_generation_tool(msg)
        

        chat = gr.ChatInterface(
            fn=chat_response,
            additional_inputs=[data_input, doc_input],
            title="💬 Asistente de Explicaciones",
            examples=[
                ["¿Qué es una regla booleana?", None, None],  # Ejemplo 1: solo mensaje
                ["Explícame la regla A & B", None, None],    # Ejemplo 2: solo mensaje
            ]
        )
        
        # Manejadores de Eventos
        run_btn.click(
            fn=workflow.run_pipeline,
            inputs=[data_input, doc_input],
            outputs=[output_panel, status]
        )
    
    return demo

# 7. Configuración y Ejecución
if __name__ == "__main__":
    load_dotenv()  # Carga las variables de entorno desde un archivo .env
    api_key = ""  # Asegúrate de tener la API key en el archivo .env
    if not api_key:
        raise ValueError("La API key de DeepSeek no está configurada en el archivo .env")
    app = create_gradio_interface(api_key)
    app.launch(server_name="0.0.0.0", server_port=7860)