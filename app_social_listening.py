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
import numpy as np  # CORREÇÃO: Importando numpy

# --- CONFIGURAÇÃO GEMINI ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("A chave da API do Google Gemini não foi encontrada. Configure-a no arquivo `secrets.toml` do seu projeto Streamlit.")
    st.stop()

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- CONSTANTE PARA LIMITE DE COMENTÁRIOS ---
MAX_COMMENTS_TO_PROCESS = 2000

# --- FUNÇÕES DE EXTRAÇÃO DE DADOS (Sem alterações) ---
@st.cache_data(show_spinner="Extraindo texto do arquivo...")
def extract_text_from_file(file_contents, file_extension):
    text_content_list = []
    try:
        if file_extension == '.csv':
            df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
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
    except Exception as e:
        st.error(f"Erro ao extrair texto do arquivo: {e}")
    return text_content_list

@st.cache_data(show_spinner=False)
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
                if len(all_comments) >= 2500:
                    break
    except Exception as e:
        st.error(f"Não foi possível baixar comentários de '{youtube_url}': {e}")
    return all_comments

# --- FUNÇÕES DE ANÁLISE COM GEMINI ---
def clean_json_response(text):
    match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    if text.strip().startswith('{') and text.strip().endswith('}'):
        return text
    return text

@st.cache_data(show_spinner="Analisando texto com a IA...")
def analyze_text_with_gemini(_text_to_analyze):
    # (Código da função mantido igual)
    if not _text_to_analyze.strip(): return None
    prompt = f"""Analise os comentários e retorne um único objeto JSON com a estrutura: {{"sentiment": {{"positive": float, "neutral": float, "negative": float, "no_sentiment_detected": float}}, "topics": [{{"name": "Nome", "positive": int, "neutral": int, "negative": int}}], "term_clusters": {{"termo1": int}}, "topic_relations": [{{"source": "Tema A", "target": "Tema B", "description": "Desc."}}]}}. Instruções: 1. `sentiment`: porcentagens, soma 100. 2. `topics`: 5-10 temas principais com contagem de sentimentos. 3. `term_clusters`: 10-20 termos significativos e frequência. 4. `topic_relations`: 3-5 pares de temas relacionados. Texto: "{_text_to_analyze}" """
    try:
        response = model.generate_content(prompt)
        data = json.loads(clean_json_response(response.text))
        return data
    except Exception as e:
        st.error(f"Erro na análise principal da IA: {e}")
        return None

# --- NOVAS FUNÇÕES PARA ANÁLISE CONTEXTUAL DE CADA GRÁFICO ---

@st.cache_data
def generate_sentiment_analysis_text(_sentiment_data):
    prompt = f"""
    Aja como um analista de dados sênior. Com base na seguinte distribuição de sentimentos: {json.dumps(_sentiment_data, indent=2)}.
    Escreva uma análise profissional de 1 a 2 parágrafos.
    - O que essa distribuição geral (positiva, negativa, neutra) sugere sobre a recepção do público?
    - Existem implicações de negócio ou de marca diretas a partir desses números? (ex: alta negatividade requer ação de gerenciamento de crise; alta positividade pode ser usada em marketing).
    - Qual o tom geral da conversa? É um público engajado, crítico, ou indiferente?
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Não foi possível gerar a análise textual para o gráfico de sentimento."

@st.cache_data
def generate_topics_analysis_text(_topics_data):
    prompt = f"""
    Aja como um estrategista de conteúdo e produto. Analise os seguintes temas e seus sentimentos associados: {json.dumps(_topics_data, indent=2, ensure_ascii=False)}.
    Escreva uma análise profissional de 1 a 2 parágrafos.
    - Quais são os 2-3 temas mais elogiados (positivos)? Como podemos amplificar esses pontos em nossa comunicação?
    - Quais são os 2-3 temas mais criticados (negativos)? Eles apontam para falhas no produto, no serviço ou na comunicação que precisam de atenção imediata?
    - Existem temas neutros com alto volume que representam oportunidades de educar ou engajar melhor o público?
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Não foi possível gerar a análise textual para o gráfico de temas."

@st.cache_data
def generate_wordcloud_analysis_text(_term_clusters_data):
    prompt = f"""
    Aja como um pesquisador de mercado (market researcher). Os seguintes termos foram os mais frequentes nos comentários: {json.dumps(list(_term_clusters_data.keys()), indent=2, ensure_ascii=False)}.
    Escreva uma análise profissional de 1 a 2 parágrafos.
    - Que história esses termos contam quando vistos em conjunto? Eles revelam o vocabulário do cliente?
    - A proeminência de certas palavras sugere o que é mais importante ("top of mind") para este público?
    - Existem jargões técnicos ou gírias que indicam um perfil de público específico (ex: iniciantes, especialistas)?
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Não foi possível gerar a análise textual para a nuvem de palavras."

@st.cache_data
def generate_relations_analysis_text(_relations_data):
    prompt = f"""
    Aja como um analista de sistemas de negócio. As seguintes relações entre temas foram identificadas: {json.dumps(_relations_data, indent=2, ensure_ascii=False)}.
    Escreva uma análise profissional de 1 a 2 parágrafos.
    - O que as conexões entre os temas revelam? Por exemplo, a conexão entre "Preço" e "Qualidade" sugere que o público está fazendo uma análise de custo-benefício.
    - Essas relações indicam uma jornada do usuário ou um processo de tomada de decisão?
    - Quais são as implicações estratégicas dessas conexões? Devemos criar conteúdo que aborde esses temas em conjunto?
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Não foi possível gerar a análise textual para o grafo de relações."


# --- OUTRAS FUNÇÕES DE GERAÇÃO (Qualitativa, Persona, etc. - Sem alterações) ---
@st.cache_data
def generate_qualitative_analysis(_analysis_results, _text_sample):
    # (Código da função mantido igual)
    prompt = f"""Como especialista em Marketing, redija uma análise qualitativa (3-4 parágrafos) com base nos dados. Foco em insights, temas e oportunidades. Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try:
        response = model.generate_content(prompt); return response.text.strip()
    except Exception as e: return f"Erro: {e}"

@st.cache_data
def generate_persona_insights(_analysis_results, _text_sample):
    # (Código da função mantido igual)
    prompt = f"""Baseado nos dados, crie uma "persona sintética". Descreva em 2-3 parágrafos: nome, dores, interesses e oportunidades. Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try:
        response = model.generate_content(prompt); return response.text.strip()
    except Exception as e: return f"Erro: {e}"

@st.cache_data
def generate_ice_score_tests(_analysis_results):
    # (Código da função mantido igual)
    prompt = f"""Como Growth Hacker, sugira 10 testes (ICE Score), ordenados. Resposta DEVE ser um único JSON. Inclua: Ordem, Nome, Descrição, Variável, Impacto (1-10), Confiança (1-10), Facilidade (1-10), ICE Score. Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try:
        response = model.generate_content(prompt); return json.loads(clean_json_response(response.text))
    except Exception as e: st.error(f"Erro: {e}"); return None

@st.cache_data
def generate_product_marketing_insights(_analysis_results):
    # (Código da função mantido igual)
    prompt = f"""Como PMM Sênior, analise os dados e crie um briefing de Product Marketing. Estruture em Markdown com: Resumo Executivo, Perfil do Público, Percepções Atuais, Desejos e Necessidades, Objeções e Barreiras, Recomendações Estratégicas (Posicionamento e Roadmap). Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try:
        response = model.generate_content(prompt); return response.text.strip()
    except Exception as e: return f"Erro: {e}"

# --- FUNÇÕES DE VISUALIZAÇÃO ---
def plot_sentiment_chart(sentiment_data):
    colors_for_pie = {'positive': '#4CAF50', 'neutral': '#1f2329', 'negative': '#ffcd03', 'no_sentiment_detected': '#cccccc'} # Positivo verde
    # (Restante da função mantido igual)
    labels_order = ['positive', 'neutral', 'negative', 'no_sentiment_detected']
    display_labels = ['Positivo', 'Neutro', 'Negativo', 'Não Detectado']
    sizes = [sentiment_data.get(label, 0.0) for label in labels_order]
    filtered_data = [(display_labels[i], sizes[i], colors_for_pie[labels_order[i]]) for i, size in enumerate(sizes) if size > 0]
    if not filtered_data: st.warning("Dados de sentimento insuficientes."); return
    filtered_labels, filtered_sizes, filtered_colors = zip(*filtered_data)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(filtered_sizes, explode=[0.03]*len(filtered_labels), labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.add_artist(plt.Circle((0,0),0.70,fc='#f3f3f3'))
    ax.axis('equal'); ax.set_title('1. Análise de Sentimento Geral', pad=18, color='#1f2329')
    st.pyplot(fig)

def plot_topics_chart(topics_data):
    if not topics_data: st.warning("Dados de temas insuficientes."); return
    df_topics = pd.DataFrame(topics_data).fillna(0)
    df_topics['Total'] = df_topics['positive'] + df_topics['neutral'] + df_topics['negative']
    df_topics = df_topics.sort_values('Total', ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_topics) * 0.5)))
    # ALTERAÇÃO DE COR: Usando verde para positivo e amarelo para negativo
    df_topics[['positive', 'neutral', 'negative']].plot(kind='barh', stacked=True, color=['#4CAF50', '#1f2329', '#ffcd03'], ax=ax)
    ax.set_title('2. Temas Mais Citados por Sentimento'); ax.set_xlabel('Número de Comentários'); ax.set_ylabel('Tema')
    ax.legend(['Positivo', 'Neutro', 'Negativo'], loc='lower right', frameon=False)
    plt.tight_layout(); st.pyplot(fig)

def plot_word_cloud(term_clusters_data):
    if not term_clusters_data: st.warning("Dados de termos insuficientes."); return
    # CORREÇÃO E ALTERAÇÃO DE COR: Usando np.random e a cor amarela
    color_func = lambda *args, **kwargs: "#ffcd03" if np.random.rand() > 0.7 else "#1f2329"
    wordcloud = WordCloud(width=700, height=400, background_color='#f3f3f3', color_func=color_func, collocations=False).generate_from_frequencies(term_clusters_data)
    fig = plt.figure(figsize=(8, 5)); plt.imshow(wordcloud, interpolation='bilinear'); plt.axis('off'); plt.title('3. Agrupamento de Termos')
    st.pyplot(fig)

def plot_topic_relations_chart(topic_relations_data):
    if not topic_relations_data: st.warning("Dados de relações insuficientes."); return
    G = nx.Graph()
    for rel in topic_relations_data:
        if rel.get('source') and rel.get('target'): G.add_edge(rel['source'], rel['target'])
    if not G.edges(): st.warning("Nenhuma relação válida encontrada."); return
    fig, ax = plt.subplots(figsize=(8, 7)); pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
    # ALTERAÇÃO DE COR: Nós principais em amarelo
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#ffcd03', alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color='#1f2329', alpha=0.6, ax=ax)
    # Trocando a cor do texto para preto para melhor legibilidade no amarelo
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='#000000', ax=ax)
    ax.set_title('4. Relação Entre Temas'); plt.axis('off'); plt.tight_layout(); st.pyplot(fig)

# --- LAYOUT DA APLICAÇÃO STREAMLIT ---
st.set_page_config(layout="wide", page_title="Social Listening Tool + AI")
st.title("🗣️ Social Listening Tool + AI")
st.markdown("---")

st.markdown(f"Carregue uma base de comentários, insira até 3 URLs de vídeos do YouTube ou cole comentários abaixo. A análise da IA será feita com uma amostra de até **{MAX_COMMENTS_TO_PROCESS} comentários**.")

# (Lógica de entrada de dados mantida igual)
if 'last_input_type' not in st.session_state: st.session_state.last_input_type = None
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload de arquivo:", type=["csv", "xls", "xlsx", "doc", "docx"], key="fileuploader")
    if uploaded_file: st.session_state.last_input_type = 'file'
with col2:
    st.write("**Ou insira até 3 URLs do YouTube:**")
    youtube_url_1 = st.text_input("URL 1:", key="yt1", placeholder="https://www.youtube.com/watch?v=...")
    youtube_url_2 = st.text_input("URL 2:", key="yt2")
    youtube_url_3 = st.text_input("URL 3:", key="yt3")
    if any([youtube_url_1, youtube_url_2, youtube_url_3]): st.session_state.last_input_type = 'youtube'
manual_text = st.text_area("Ou cole os comentários (um por linha):", key="manual", height=150)
if manual_text: st.session_state.last_input_type = 'manual'

all_comments_list = []
if st.session_state.last_input_type == 'file' and uploaded_file:
    all_comments_list = extract_text_from_file(uploaded_file.read(), os.path.splitext(uploaded_file.name)[1].lower())
elif st.session_state.last_input_type == 'youtube':
    youtube_urls = [url.strip() for url in [youtube_url_1, youtube_url_2, youtube_url_3] if url.strip()]
    if youtube_urls:
        temp_comments = []
        for url in youtube_urls:
            temp_comments.extend(download_youtube_comments(url))
        all_comments_list = temp_comments
elif st.session_state.last_input_type == 'manual' and manual_text:
    all_comments_list = [line for line in manual_text.split("\n") if line.strip()]

if all_comments_list:
    original_comment_count = len(all_comments_list)
    comments_for_analysis = all_comments_list[:MAX_COMMENTS_TO_PROCESS]
    st.success(f"{original_comment_count} comentários carregados. Analisando uma amostra de {len(comments_for_analysis)}.")
    text_to_analyze = "\n".join(comments_for_analysis)
    analysis_results = analyze_text_with_gemini(text_to_analyze)

    if analysis_results:
        tabs = st.tabs(["📊 Sentimento", "💡 Temas", "🔑 Termos-Chave", "🔗 Relações", "📝 Análise Qualitativa", "🧑‍💼 Persona", "🚀 Growth", "📈 PMM"])

        with tabs[0]:
            plot_sentiment_chart(analysis_results.get('sentiment', {}))
            with st.spinner("Gerando análise do gráfico..."):
                st.markdown(generate_sentiment_analysis_text(analysis_results.get('sentiment', {})))
        with tabs[1]:
            plot_topics_chart(analysis_results.get('topics', []))
            with st.spinner("Gerando análise do gráfico..."):
                st.markdown(generate_topics_analysis_text(analysis_results.get('topics', [])))
        with tabs[2]:
            plot_word_cloud(analysis_results.get('term_clusters', {}))
            with st.spinner("Gerando análise do gráfico..."):
                st.markdown(generate_wordcloud_analysis_text(analysis_results.get('term_clusters', {})))
        with tabs[3]:
            plot_topic_relations_chart(analysis_results.get('topic_relations', []))
            with st.spinner("Gerando análise do gráfico..."):
                st.markdown(generate_relations_analysis_text(analysis_results.get('topic_relations', [])))
        with tabs[4]:
            with st.spinner("Gerando análise qualitativa..."):
                st.markdown(generate_qualitative_analysis(analysis_results, text_to_analyze))
        with tabs[5]:
            with st.spinner("Gerando insights de persona..."):
                st.markdown(generate_persona_insights(analysis_results, text_to_analyze))
        with tabs[6]:
            with st.spinner("Gerando testes de growth..."):
                ice = generate_ice_score_tests(analysis_results)
                if ice: st.dataframe(pd.DataFrame(ice), hide_index=True, use_container_width=True)
        with tabs[7]:
            with st.spinner("Gerando insights de Product Marketing..."):
                st.markdown(generate_product_marketing_insights(analysis_results))
    else:
        st.error("A análise com a IA falhou. Verifique os dados ou tente novamente.")
else:
    st.info("Aguardando dados para iniciar a análise...")

# --- FOOTER ---
st.markdown("---")
st.subheader("💡 Gostou da aplicação?")
st.markdown("Para acessar a versão completa e sem limitações, clique no botão abaixo.")
st.link_button("Acessar The Research AI", "https://www.theresearchai.online/", type="primary")
st.markdown("---")
st.markdown("Desenvolvido por Pedro Costa")
