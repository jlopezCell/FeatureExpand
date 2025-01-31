import os
import pandas as pd
import sympy as sp
import gradio as gr
from crewai import Agent, Task, Crew
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Sistema de Gestión Documental
class DocumentAnalyzer:
    def __init__(self):
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
    
    def upload_document(self, file_path):
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            self.vector_store = FAISS.from_documents(pages, self.embeddings)
            return "✅ Documentos analizados exitosamente!"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def query_documents(self, question):
        if not self.vector_store:
            return "Primero cargue documentos técnicos"
        docs = self.vector_store.similarity_search(question, k=3)
        return "\n\n".join([f"📄 Página {d.metadata['page']}:\n{d.page_content}" for d in docs])

# 2. Sistema de Agentes Especializados
class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description
        
# 2. Sistema de Agentes Especializados
class MLAgents:
    def __init__(self, document_analyzer):
        self.document_analyzer = document_analyzer
        self.llm = ChatOpenAI(model="gpt-4-1106-preview")
        
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
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, mean_squared_error(y_test, model.predict(X_test))
    
    def explanation_generation_tool(self, rule):
        context = self.document_analyzer.query_documents(rule)
        prompt = f"""
        Contexto del dominio:
        {context}
        
        Explica esta regla booleana para no expertos: {rule}
        """
        return self.llm.invoke(prompt).content


# 3. Flujo de Trabajo Principal
class MLWorkflow:
    def __init__(self):
        self.document_analyzer = DocumentAnalyzer()
        self.agents = MLAgents(self.document_analyzer)
        self.current_project = None
    
    def run_pipeline(self, data, documents=None):
        try:
            # Procesar documentos
            if documents:
                doc_status = self.document_analyzer.upload_document(documents.name)
                if "❌" in doc_status:
                    return doc_status
                
            # Generar características
            feature_task = Task(
                description=f"Generar características booleanas para los datos",
                agent=self.agents.agents['feature_engineer'],
                expected_output="Lista de reglas booleanas simplificadas"
            )
            
            # Validar características
            validation_task = Task(
                description="Validar reglas con conocimiento del dominio",
                agent=self.agents.agents['domain_validator'],
                expected_output="Reglas validadas con documentos técnicos",
                context=[feature_task]
            )
            
            # Optimizar modelo
            optimization_task = Task(
                description="Entrenar y optimizar modelo lineal",
                agent=self.agents.agents['model_optimizer'],
                expected_output="Modelo entrenado con métricas de rendimiento",
                context=[validation_task]
            )
            
            # Generar explicaciones
            explanation_task = Task(
                description="Crear explicaciones contextuales para cada regla",
                agent=self.agents.agents['explanation_expert'],
                expected_output="Reporte explicativo completo",
                context=[optimization_task]
            )
            
            crew = Crew(
                agents=list(self.agents.agents.values()),
                tasks=[feature_task, validation_task, optimization_task, explanation_task],
                verbose=2
            )
            
            return crew.kickoff()
        
        except Exception as e:
            return f"❌ Error en el pipeline: {str(e)}"

# 4. Interfaz de Usuario
def create_gradio_interface():
    workflow = MLWorkflow()
    
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
        chat = gr.ChatInterface(
            lambda msg, history: workflow.agents.explanation_generation_tool(msg),
            additional_inputs=[data_input, doc_input],
            title="💬 Asistente de Explicaciones"
        )
        
        # Manejadores de Eventos
        run_btn.click(
            fn=workflow.run_pipeline,
            inputs=[data_input, doc_input],
            outputs=output_panel
        )
    
    return demo

# Configuración y Ejecución
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "sk-tu-api-key-aqui"  # Reemplazar con tu API key
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)