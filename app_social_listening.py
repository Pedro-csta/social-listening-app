import streamlit as st
import pandas as pd
import json
import io
from docx import Document
from youtube_comment_downloader import YoutubeCommentDownloader
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import os
import google.generativeai as genai

# --- CONFIGURAÇÃO GEMINI ---
# Tenta carregar a chave da API dos secrets do Streamlit
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("A chave da API do Google Gemini não foi encontrada. Configure-a no arquivo `secrets.toml` do seu projeto Streamlit.")
    st.stop()

# Configura a API do Gemini
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash') # Modelo atualizado para versão mais recente

# --- CONSTANTE PARA LIMITE DE COMENTÁRIOS ---
MAX_COMMENTS_TO_PROCESS = 2000 # Limite de comentários a serem enviados para a IA

# --- FUNÇÕES DE EXTRAÇÃO DE DADOS ---
@st.cache_data(show_spinner="Extraindo texto do arquivo...")
def extract_text_from_file(file_contents, file_extension):
    text_content_list = []
    try:
        if file_extension == '.csv':
            df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            # Tenta pegar a coluna 'comentario' ou junta todas as colunas
            if 'comentario' in df.columns:
                text_content_list = df['comentario'].dropna().astype(str).tolist()
            else:
                text_content_list = [" ".join(row.dropna().astype(str)) for _, row in df.iterrows()]
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(io.BytesIO(file_contents))
            if 'comentario' in df.columns:
                text_content_list = df['comentario'].dropna().astype(str).tolist()
            else:
                text_content_list = [" ".join(row.dropna().astype(str)) for _, row in df.iterrows()]
        elif file_extension in ['.doc', '.docx']:
            document = Document(io.BytesIO(file_contents))
            text_content_list = [p.text for p in document.paragraphs if p.text.strip()]
        else:
            st.warning(f"Formato de arquivo não suportado: {file_extension}.")
    except Exception as e:
        st.error(f"Erro ao extrair texto do arquivo: {e}")
    return text_content_list

@st.cache_data(show_spinner=False) # Spinner é controlado manualmente dentro da função
def download_youtube_comments(youtube_url):
    all_comments = []
    try:
        downloader = YoutubeCommentDownloader()
        match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if not match:
            st.error(f"URL do YouTube inválida: {youtube_url}.")
            return []
        
        video_id = match.group(1)
        with st.spinner(f"Baixando comentários de: {youtube_url}..."):
            comments_generator = downloader.get_comments(video_id)
            for comment in comments_generator:
                if comment and 'text' in comment:
                    all_comments.append(comment['text'])
                # Limite de segurança para não baixar infinitamente em vídeos muito populares
                if len(all_comments) >= 2500:
                    st.info(f"Limite de download atingido para o vídeo {youtube_url}.")
                    break
    except Exception as e:
        st.error(f"Não foi possível baixar comentários de '{youtube_url}': {e}")
    return all_comments

# --- FUNÇÕES DE ANÁLISE COM GEMINI ---
def clean_json_response(text):
    """Função para limpar a resposta do modelo e extrair o JSON."""
    match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    if text.strip().startswith('{') and text.strip().endswith('}'):
        return text
    return text

@st.cache_data(show_spinner="Analisando texto com a IA...")
def analyze_text_with_gemini(_text_to_analyze): # Adicionado _ para indicar que é um parâmetro "mutável"
    if not _text_to_analyze.strip():
        return None
    
    num_comments_in_prompt = len(_text_to_analyze.split('\n'))
    
    prompt = f"""
    Analise o texto de {num_comments_in_prompt} comentários de redes sociais de forma objetiva e consistente.
    Sua resposta DEVE ser um único objeto JSON, sem nenhum texto ou formatação adicional fora dele.

    O JSON deve seguir EXATAMENTE esta estrutura:
    {{
      "sentiment": {{"positive": float, "neutral": float, "negative": float, "no_sentiment_detected": float}},
      "topics": [{{"name": "Nome do Tema", "positive": int, "neutral": int, "negative": int}}],
      "term_clusters": {{"termo1": int, "termo2": int}},
      "topic_relations": [{{"source": "Tema A", "target": "Tema B", "description": "Descrição da relação"}}]
    }}

    Instruções:
    1.  **sentiment**: Calcule a porcentagem de comentários 'Positivos', 'Neutros', 'Negativos' e 'Sem Sentimento Detectado'. A soma DEVE ser 100.
    2.  **topics**: Liste de 5 a 10 temas principais. Para cada um, forneça a contagem EXATA de comentários por sentimento.
    3.  **term_clusters**: Liste os 10 a 20 termos mais significativos e sua contagem. Não inclua palavras de parada.
    4.  **topic_relations**: Identifique de 3 a 5 pares de temas que aparecem juntos com frequência e descreva a relação.

    Texto para análise:
    "{_text_to_analyze}"
    """
    try:
        response = model.generate_content(prompt)
        response_text = clean_json_response(response.text)
        data = json.loads(response_text)
        
        # Garante que a soma do sentimento seja 100%
        if 'sentiment' in data and isinstance(data['sentiment'], dict):
            if 'no_sentiment_detected' not in data['sentiment']:
                total = sum(v for k, v in data['sentiment'].items() if k in ['positive', 'neutral', 'negative'])
                data['sentiment']['no_sentiment_detected'] = max(0, 100.0 - total)
            
            total_sum = sum(data['sentiment'].values())
            if total_sum > 0 and total_sum != 100:
                for key in data['sentiment']:
                    data['sentiment'][key] = round((data['sentiment'][key] / total_sum) * 100, 2)
        return data
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Erro ao processar a resposta da IA: {e}")
        st.code(f"Resposta recebida: {response.text if 'response' in locals() else 'N/A'}")
        return None

@st.cache_data
def generate_qualitative_analysis(_analysis_results, _text_sample):
    prompt = f"""
    Como especialista em Marketing, redija uma análise qualitativa (3-4 parágrafos) com base nos dados de social listening. Foco em insights estratégicos, temas relevantes e oportunidades.
    Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Erro ao gerar análise qualitativa: {e}"

@st.cache_data
def generate_persona_insights(_analysis_results, _text_sample):
    prompt = f"""
    Baseado nos dados, crie um insight de "persona sintética". Descreva em 2-3 parágrafos: um nome para a persona, suas dores, interesses e oportunidades de engajamento.
    Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Erro ao gerar insights de persona: {e}"

@st.cache_data
def generate_ice_score_tests(_analysis_results):
    prompt = f"""
    Como Growth Hacker, sugira 10 testes de Growth usando a metodologia ICE Score, ordenados do maior para o menor score.
    A resposta DEVE ser um único objeto JSON (uma lista de testes) e nada mais.
    Para cada teste, inclua: "Ordem", "Nome do Teste", "Descrição do Teste", "Variável de Alavancagem" (Canal, Segmentação, Formato, Criativo ou Copy/Argumento), "Impacto (1-10)", "Confiança (1-10)", "Facilidade (1-10)", e "ICE Score" ((I+C+E)/3).
    Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try:
        response = model.generate_content(prompt)
        response_text = clean_json_response(response.text)
        return json.loads(response_text)
    except (json.JSONDecodeError, Exception) as e:
        st.error(f"Erro ao gerar testes ICE: {e}")
        st.code(f"Resposta recebida: {response.text if 'response' in locals() else 'N/A'}")
        return None

# --- FUNÇÕES DE VISUALIZAÇÃO ---
def plot_sentiment_chart(sentiment_data):
    labels_order = ['positive', 'neutral', 'negative', 'no_sentiment_detected']
    display_labels = ['Positivo', 'Neutro', 'Negativo', 'Não Detectado']
    colors_for_pie = {'positive': '#ff99b0', 'neutral': '#1f2329', 'negative': '#fe1874', 'no_sentiment_detected': '#cccccc'}
    sizes = [sentiment_data.get(label, 0.0) for label in labels_order]
    filtered_data = [(display_labels[i], sizes[i], colors_for_pie[labels_order[i]]) for i, size in enumerate(sizes) if size > 0]
    if not filtered_data:
        st.warning("Dados de sentimento insuficientes para gerar gráfico.")
        return
    filtered_labels, filtered_sizes, filtered_colors = zip(*filtered_data)
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(filtered_sizes, explode=[0.03]*len(filtered_labels), labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    for autotext in autotexts:
        autotext.set_color('#FFFFFF'); autotext.set_fontsize(12); autotext.set_fontweight('bold')
    for text in texts:
        text.set_color('#1f2329'); text.set_fontsize(10)
    ax.add_artist(plt.Circle((0,0),0.70,fc='#f3f3f3'))
    ax.axis('equal'); ax.set_title('1. Análise de Sentimento Geral', pad=18, color='#1f2329')
    st.pyplot(fig)

def plot_topics_chart(topics_data):
    if not topics_data:
        st.warning("Dados de temas insuficientes para gerar gráfico.")
        return
    df_topics = pd.DataFrame(topics_data).fillna(0)
    df_topics['Total'] = df_topics['positive'] + df_topics['neutral'] + df_topics['negative']
    df_topics = df_topics.sort_values('Total', ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_topics) * 0.5)))
    df_topics[['positive', 'neutral', 'negative']].plot(kind='barh', stacked=True, color=['#ff99b0', '#1f2329', '#fe1874'], ax=ax)
    ax.set_title('2. Temas Mais Citados por Sentimento', color='#1f2329'); ax.set_xlabel('Número de Comentários', color='#1f2329'); ax.set_ylabel('Tema', color='#1f2329')
    ax.set_yticklabels(df_topics['name'], color='#1f2329'); ax.tick_params(axis='x', colors='#1f2329'); ax.tick_params(axis='y', colors='#1f2329')
    ax.legend(['Positivo', 'Neutro', 'Negativo'], loc='lower right', frameon=False, labelcolor='#1f2329')
    plt.tight_layout(); st.pyplot(fig)

def plot_word_cloud(term_clusters_data):
    if not term_clusters_data:
        st.warning("Dados de termos insuficientes para nuvem de palavras.")
        return
    wordcloud = WordCloud(width=700, height=400, background_color='#f3f3f3', color_func=lambda *args, **kwargs: "#fe1874" if pd.np.random.rand() > 0.7 else "#1f2329", min_font_size=12, max_words=60, prefer_horizontal=0.8, collocations=False).generate_from_frequencies(term_clusters_data)
    fig = plt.figure(figsize=(8, 5)); plt.imshow(wordcloud, interpolation='bilinear'); plt.axis('off'); plt.title('3. Agrupamento de Termos (Nuvem de Palavras)', pad=16, fontsize=15)
    st.pyplot(fig)

def plot_topic_relations_chart(topic_relations_data):
    if not topic_relations_data:
        st.warning("Dados insuficientes para grafo de relações.")
        return
    G = nx.Graph()
    for rel in topic_relations_data:
        if rel.get('source') and rel.get('target'):
            G.add_edge(rel['source'], rel['target'], description=rel.get('description'))
    if not G.edges():
        st.warning("Nenhuma relação válida para construir o grafo.")
        return
    fig, ax = plt.subplots(figsize=(8, 7)); pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#fe1874', alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color='#1f2329', alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='#FFFFFF', ax=ax) # Rótulos brancos para contraste
    ax.set_title('4. Relação Entre Temas (Grafo de Rede)', pad=16, color='#1f2329'); plt.axis('off'); plt.tight_layout(); st.pyplot(fig)

# --- LAYOUT DA APLICAÇÃO STREAMLIT ---
st.set_page_config(layout="wide", page_title="Social Listening Tool + AI")
st.title("🗣️ Social Listening Tool + AI")
st.markdown("---")

st.markdown(f"Carregue uma base de comentários, insira até 3 URLs de vídeos do YouTube ou cole comentários abaixo. **A análise da IA será feita com uma amostra de até {MAX_COMMENTS_TO_PROCESS} comentários.**")

all_comments_list = []
original_comment_count = 0

# --- LÓGICA DE ENTRADA DE DADOS ---
# Limpa o estado da sessão se um novo tipo de input for usado
if 'last_input_type' not in st.session_state:
    st.session_state.last_input_type = None

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader(
        "Faça upload de um arquivo (.csv, .xls, .xlsx, .doc, .docx):",
        type=["csv", "xls", "xlsx", "doc", "docx"],
        key="fileuploader"
    )
    if uploaded_file:
        st.session_state.last_input_type = 'file'

with col2:
    st.write("**Ou insira até 3 URLs de vídeos do YouTube:**")
    youtube_url_1 = st.text_input("URL do Vídeo 1:", key="yt1")
    youtube_url_2 = st.text_input("URL do Vídeo 2:", key="yt2")
    youtube_url_3 = st.text_input("URL do Vídeo 3:", key="yt3")
    
    # Verifica se alguma URL foi inserida
    if any([youtube_url_1, youtube_url_2, youtube_url_3]):
        st.session_state.last_input_type = 'youtube'

manual_text = st.text_area("Ou cole os comentários aqui (um por linha):", key="manual")
if manual_text:
    st.session_state.last_input_type = 'manual'


# --- PROCESSAMENTO DOS DADOS DE ENTRADA ---
if st.session_state.last_input_type == 'file' and uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    file_contents = uploaded_file.read()
    all_comments_list = extract_text_from_file(file_contents, file_extension)

elif st.session_state.last_input_type == 'youtube':
    # Coleta todas as URLs válidas
    youtube_urls = [url.strip() for url in [youtube_url_1, youtube_url_2, youtube_url_3] if url.strip()]
    if youtube_urls:
        temp_comments = []
        # Itera sobre cada URL, baixa os comentários e os adiciona a uma lista temporária
        for url in youtube_urls:
            comments_from_url = download_youtube_comments(url)
            if comments_from_url:
                temp_comments.extend(comments_from_url)
        all_comments_list = temp_comments

elif st.session_state.last_input_type == 'manual' and manual_text:
    all_comments_list = [line for line in manual_text.split("\n") if line.strip()]

# --- LÓGICA DE ANÁLISE E EXIBIÇÃO ---
if all_comments_list:
    original_comment_count = len(all_comments_list)
    
    # Aplica o limite de comentários para análise
    comments_for_analysis = all_comments_list[:MAX_COMMENTS_TO_PROCESS]
    
    # Exibe mensagem de sucesso com a contagem correta
    if original_comment_count > MAX_COMMENTS_TO_PROCESS:
        st.success(f"{original_comment_count} comentários carregados! A análise será feita em uma amostra de {len(comments_for_analysis)} comentários para otimizar o processamento.")
    else:
        st.success(f"{len(comments_for_analysis)} comentários carregados e prontos para análise!")

    # Inicia a análise
    text_to_analyze = "\n".join(comments_for_analysis)
    analysis_results = analyze_text_with_gemini(text_to_analyze)

    if analysis_results:
        # Cria as abas para exibir os resultados
        tabs = st.tabs([
            "📊 Sentimento", "💡 Temas", "🔑 Termos-Chave", "🔗 Relações",
            "📝 Análise Qualitativa", "🧑‍💼 Persona Sintética", "🚀 Testes de Growth"
        ])

        with tabs[0]:
            plot_sentiment_chart(analysis_results.get('sentiment', {}))
            with st.expander("Ver dados brutos"): st.json(analysis_results.get('sentiment', {}))

        with tabs[1]:
            plot_topics_chart(analysis_results.get('topics', []))
            with st.expander("Ver dados brutos"): st.json(analysis_results.get('topics', []))

        with tabs[2]:
            plot_word_cloud(analysis_results.get('term_clusters', {}))
            with st.expander("Ver dados brutos"): st.json(analysis_results.get('term_clusters', {}))

        with tabs[3]:
            plot_topic_relations_chart(analysis_results.get('topic_relations', []))
            with st.expander("Ver dados brutos"): st.json(analysis_results.get('topic_relations', []))

        with tabs[4]:
            with st.spinner("Gerando análise qualitativa..."):
                qualitative = generate_qualitative_analysis(analysis_results, text_to_analyze)
                st.markdown(qualitative)

        with tabs[5]:
            with st.spinner("Gerando insights de persona..."):
                persona = generate_persona_insights(analysis_results, text_to_analyze)
                st.markdown(persona)

        with tabs[6]:
            with st.spinner("Gerando sugestões de testes de growth..."):
                ice = generate_ice_score_tests(analysis_results)
                if ice:
                    df_ice = pd.DataFrame(ice)
                    # Garante a ordem e existência das colunas
                    cols_order = ["Ordem", "Nome do Teste", "Descrição do Teste", "Variável de Alavancagem", "Impacto (1-10)", "Confiança (1-10)", "Facilidade (1-10)", "ICE Score"]
                    df_ice = df_ice[[c for c in cols_order if c in df_ice.columns]]
                    if "ICE Score" in df_ice.columns:
                        df_ice = df_ice.sort_values(by="ICE Score", ascending=False)
                    st.dataframe(df_ice, hide_index=True, use_container_width=True)
                    with st.expander("Ver dados brutos"): st.json(ice)
                else:
                    st.warning("Não foi possível gerar sugestões de testes de growth.")
    else:
        st.error("A análise com a IA falhou. Verifique as mensagens de erro acima ou tente novamente.")

else:
    st.info("Aguardando dados para iniciar a análise...")

# --- FOOTER ---
st.markdown("---")
st.subheader("💡 Gostou de testar a aplicação?")
st.markdown("""
Essa versão de teste possuí uma limitação de comentários que podem ser analisados e de volume de análises por dia.
Caso tenha interesse em acessar a aplicação completa, clique no botão abaixo para conhecer a versão final.
""")

TALLY_FORM_URL = "https://www.theresearchai.online/"
st.link_button("Clique aqui e acesse a ferramenta!", TALLY_FORM_URL, type="primary")

st.markdown("---")
st.markdown("Desenvolvido com Python, ❤️ e AI por Pedro Costa | Product Marketing & Martech Specialist")
