import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client
from botbuilder.core import BotFrameworkAdapter, TurnContext, BotFrameworkAdapterSettings
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.schema import Activity
from aiohttp import web
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

# Carregar variáveis de ambiente
load_dotenv()

# Função auxiliar para garantir que variáveis de ambiente não sejam None
def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"A variável de ambiente {name} não está definida.")
    return value

# Configurar clientes
openai_client = AsyncOpenAI(api_key=get_env_var("OPENAI_API_KEY"))
supabase: Client = create_client(
    get_env_var("SUPABASE_URL"),
    get_env_var("SUPABASE_KEY")
)

# Configurar o adaptador do bot
settings = BotFrameworkAdapterSettings(
    app_id=get_env_var("MICROSOFT_APP_ID"),
    app_password=get_env_var("MICROSOFT_APP_PASSWORD")
)
adapter = BotFrameworkAdapter(settings)

# Classe do bot
class CruxBot:
    async def on_turn(self, turn_context: TurnContext):
        if turn_context.activity.type == "message":
            # Obter a pergunta do usuário
            user_message = turn_context.activity.text

            try:
                # Gerar embedding da pergunta
                embedding_response = await openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=user_message
                )
                embedding = embedding_response.data[0].embedding

                # Buscar documento relevante no Supabase
                response = supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': embedding,
                        'match_threshold': 0.7,
                        'match_count': 1
                    }
                ).execute()

                # Preparar o contexto
                context = ""
                if response.data and len(response.data) > 0:
                    context = response.data[0]['content']

                # Montar o prompt final
                system_prompt = """Você é o Crux, um especialista em ajudar pessoas com suas dúvidas. \nUse o contexto fornecido para responder de forma precisa e útil."""

                messages = [
                    ChatCompletionSystemMessageParam(role="system", content=system_prompt),
                    ChatCompletionUserMessageParam(role="user", content=f"Contexto: {context}\n\nPergunta: {user_message}")
                ]

                # Obter resposta da OpenAI
                completion = await openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                resposta = completion.choices[0].message.content or "Desculpe, não consegui gerar uma resposta."
                # Enviar resposta para o usuário
                await turn_context.send_activity(str(resposta))
            except Exception as e:
                await turn_context.send_activity(f"Desculpe, ocorreu um erro: {str(e)}")

# Criar instância do bot
bot = CruxBot()

# Configurar o servidor web
async def messages(req: web.Request) -> web.Response:
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers["Authorization"] if "Authorization" in req.headers else ""

    response = await adapter.process_activity(activity, auth_header, bot.on_turn)
    if response:
        return web.json_response(data=response.body, status=response.status)
    return web.Response(status=201)

# Configurar o aplicativo
app = web.Application(middlewares=[aiohttp_error_middleware])
app.router.add_post("/api/messages", messages)

# Iniciar o servidor
if __name__ == "__main__":
    print("Iniciando servidor na porta 3978...")
    web.run_app(app, host="localhost", port=3978)
