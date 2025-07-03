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
import numpy as np

# --- CONFIGURAÇÃO GEMINI ---
gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
if not gemini_api_key:
    st.error("A chave da API do Google Gemini não foi encontrada. Configure-a no arquivo `secrets.toml` do seu projeto Streamlit.")
    st.stop()

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- CONSTANTE PARA LIMITE DE COMENTÁRIOS ---
MAX_COMMENTS_TO_PROCESS = 1000

# --- FUNÇÕES DE EXTRAÇÃO DE DADOS ---
@st.cache_data(show_spinner="Extraindo texto do arquivo...")
def extract_text_from_file(file_contents, file_extension):
    text_content_list = []
    try:
        if file_extension == '.csv':
            df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            if 'comentario' in df.columns: text_content_list = df['comentario'].dropna().astype(str).tolist()
            else: text_content_list = [" ".join(row.dropna().astype(str)) for _, row in df.iterrows()]
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(io.BytesIO(file_contents))
            if 'comentario' in df.columns: text_content_list = df['comentario'].dropna().astype(str).tolist()
            else: text_content_list = [" ".join(row.dropna().astype(str)) for _, row in df.iterrows()]
        elif file_extension in ['.doc', '.docx']:
            document = Document(io.BytesIO(file_contents))
            text_content_list = [p.text for p in document.paragraphs if p.text.strip()]
    except Exception as e: st.error(f"Erro ao extrair texto do arquivo: {e}")
    return text_content_list

@st.cache_data(show_spinner=False)
def download_youtube_comments(youtube_url):
    all_comments = []
    try:
        downloader = YoutubeCommentDownloader()
        match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if not match: st.error(f"URL do YouTube inválida: {youtube_url}."); return []
        video_id = match.group(1)
        with st.spinner(f"Baixando comentários de: {youtube_url}..."):
            comments_generator = downloader.get_comments(video_id)
            for comment in comments_generator:
                if comment and 'text' in comment: all_comments.append(comment['text'])
                if len(all_comments) >= 2500: break
    except Exception as e: st.error(f"Não foi possível baixar comentários de '{youtube_url}': {e}")
    return all_comments

# --- FUNÇÕES DE ANÁLISE COM GEMINI ---
def clean_json_response(text):
    match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r'^\s*\{[\s\S]*?\}\s*$', text, re.DOTALL)
    if match: return match.group(0)
    return text

@st.cache_data(show_spinner="Analisando texto com a IA...")
def analyze_text_with_gemini(_text_to_analyze):
    if not _text_to_analyze.strip(): return None
    prompt = f"""Analise os comentários e retorne um único objeto JSON com a estrutura: {{"sentiment": {{"positive": float, "neutral": float, "negative": float, "no_sentiment_detected": float}}, "topics": [{{"name": "Nome", "positive": int, "neutral": int, "negative": int}}], "term_clusters": {{"termo1": int}}, "topic_relations": [{{"source": "Tema A", "target": "Tema B", "description": "Desc."}}]}}. Instruções: 1. `sentiment`: porcentagens, soma 100. 2. `topics`: 5-10 temas principais com contagem de sentimentos. 3. `term_clusters`: 10-20 termos significativos e frequência. 4. `topic_relations`: 3-5 pares de temas relacionados. Texto: "{_text_to_analyze}" """
    try:
        response = model.generate_content(prompt)
        data = json.loads(clean_json_response(response.text))
        return data
    except Exception as e: st.error(f"Erro na análise principal da IA: {e}"); return None

# --- FUNÇÕES DE ANÁLISE CONTEXTUAL ---
@st.cache_data
def generate_sentiment_analysis_text(_sentiment_data):
    prompt = f"""Aja como um analista de dados sênior. Com base na seguinte distribuição de sentimentos: {json.dumps(_sentiment_data)}. Escreva uma análise profissional de 1 a 2 parágrafos sobre o que essa distribuição sugere sobre a recepção do público, as implicações de negócio e o tom geral da conversa."""
    try: response = model.generate_content(prompt); return response.text.strip()
    except Exception: return "Não foi possível gerar a análise textual para o gráfico de sentimento."

@st.cache_data
def generate_topics_analysis_text(_topics_data):
    prompt = f"""Aja como um Analista de Mercado e Estrategista de Negócios Sênior. Sua tarefa é transformar os dados brutos de temas e sentimentos em um memorando estratégico coeso e acionável para as equipes de marketing e produto. Dados: {json.dumps(_topics_data, indent=2, ensure_ascii=False)}. Escreva um texto corrido e natural, sem quebras por tópicos ou bullet points. A análise deve fluir como um relatório coeso, abordando: 1. Síntese Estratégica: A história central que os temas contam. 2. Análise dos Pontos Fortes: Temas positivos e como o marketing pode alavancá-los. 3. Análise dos Desafios e Oportunidades: Temas negativos/neutros e o que revelam para o produto. 4. Recomendação Final: Onde focar a atenção estratégica no curto prazo. O objetivo é que este texto seja 'pronto para enviar' para um gestor de produto ou marketing."""
    try: response = model.generate_content(prompt); return response.text.strip()
    except Exception: return "Não foi possível gerar a análise textual para o gráfico de temas."

@st.cache_data
def generate_wordcloud_analysis_text(_term_clusters_data):
    prompt = f"""Aja como um pesquisador de mercado. Os seguintes termos foram os mais frequentes nos comentários: {json.dumps(list(_term_clusters_data.keys()))}. Escreva uma análise profissional de 1 a 2 parágrafos sobre a história que esses termos contam, o que é "top of mind" para o público e o que o vocabulário revela sobre seu perfil."""
    try: response = model.generate_content(prompt); return response.text.strip()
    except Exception: return "Não foi possível gerar a análise textual para a nuvem de palavras."

@st.cache_data
def generate_relations_analysis_text(_relations_data):
    prompt = f"""Aja como um analista de sistemas de negócio. As seguintes relações entre temas foram identificadas: {json.dumps(_relations_data)}. Escreva uma análise profissional de 1 a 2 parágrafos sobre o que as conexões entre os temas revelam sobre a jornada do usuário e quais as implicações estratégicas disso."""
    try: response = model.generate_content(prompt); return response.text.strip()
    except Exception: return "Não foi possível gerar a análise textual para o grafo de relações."

@st.cache_data
def generate_qualitative_analysis(_analysis_results, _text_sample):
    prompt = f"""Como especialista em Marketing, redija uma análise qualitativa (3-4 parágrafos) com base nos dados. Foco em insights, temas e oportunidades. Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try: response = model.generate_content(prompt); return response.text.strip()
    except Exception as e: return f"Erro: {e}"

# --- PROMPT ATUALIZADO COM EXEMPLO PARA A PERSONA SINTÉTICA ---
@st.cache_data
def generate_persona_insights(_analysis_results, _text_sample):
    prompt = f"""
    Aja como um Estrategista de Marketing e Produto de classe mundial, especialista em destilar dados brutos em perfis de persona acionáveis.

    Sua tarefa é criar um perfil de persona sintética extremamente detalhado e estratégico, com base nos dados de social listening fornecidos.

    **Dados da Análise:**
    {json.dumps(_analysis_results, ensure_ascii=False)}

    ---

    **EXEMPLO DE ESTRUTURA E QUALIDADE PERFEITA:**

    **Persona Identificada:** O Investidor Global Consciente

    **1. Perfil Resumido:**
    O Investidor Global Consciente é um profissional ou pessoa com renda disponível que busca diversificar seus investimentos internacionalmente, priorizando a otimização de custos e a transparência. Ele é pragmático e busca informações detalhadas antes de tomar decisões, demonstrando preocupação com impostos e regulamentações.

    **2. Dores e Necessidades Principais:**
    * Taxas e Custos elevados: A principal preocupação é com taxas de corretagem, IOF, e spread, buscando soluções mais econômicas.
    * Complexidade da Declaração de Imposto de Renda sobre investimentos no exterior: Falta clareza e facilidade no processo de declaração de ganhos de capital e dividendos obtidos no exterior.
    * Falta de transparência e suporte adequado de algumas corretoras: Experiências negativas com o suporte ao cliente e dificuldades em resolver problemas com corretoras geram desconfiança.

    **3. Desejos e Motivações:**
    * Investimentos no exterior com baixo custo: Busca plataformas e estratégias para investir internacionalmente com as menores taxas e custos possíveis.
    * Acesso a informações claras e precisas: Necessita de informações detalhadas e transparentes sobre investimentos, impostos e regulamentações.
    * Diversificação de portfólio com ETFs: Interesse em ETFs como IVV e IVVB11 para diversificação internacional.

    **4. Tom de Voz e Comportamento:**
    O Investidor Global Consciente se comunica de forma pragmática e direta, valorizando informações objetivas e dados quantitativos. Ele é cético em relação a promessas exageradas e busca comprovação de resultados. Realiza pesquisas detalhadas antes de tomar decisões, comparando diversas opções e lendo avaliações de outros usuários.

    **5. Oportunidades de Engajamento:**
    * Desenvolver um guia completo e atualizado sobre a declaração de imposto de renda para investimentos no exterior, com checklists e exemplos práticos.
    * Lançar uma calculadora online que simule os custos de investimento em diferentes plataformas, incluindo taxas, IOF e spread.
    * Criar webinars e workshops gratuitos com especialistas em investimentos internacionais e planejamento tributário.

    ---

    **Sua Tarefa Agora:**
    Com base nos dados da análise fornecidos no início deste prompt, gere um novo perfil de persona, seguindo **EXATAMENTE** a mesma estrutura, formato e nível de profundidade do exemplo acima.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Erro ao gerar insights de persona: {e}"


@st.cache_data
def generate_ice_score_tests(_analysis_results):
    prompt = f"""Como Growth Hacker, sugira 10 testes (ICE Score), ordenados. Resposta DEVE ser um único JSON. Inclua: Ordem, Nome, Descrição, Variável, Impacto (1-10), Confiança (1-10), Facilidade (1-10), ICE Score. Dados: {json.dumps(_analysis_results, ensure_ascii=False)}"""
    try: response = model.generate_content(prompt); return json.loads(clean_json_response(response.text))
    except Exception as e: st.error(f"Erro ao gerar testes ICE: {e}"); return None

@st.cache_data
def generate_product_marketing_insights(_analysis_results):
    prompt = f"""
    Aja como um Consultor de Estratégia de Produto e Marketing de alto nível. Sua tarefa é destilar os dados de social listening em um briefing executivo, conciso e extremamente acionável. A filosofia é 'menos é mais', mas com profundidade. Evite repetições de outras seções e foque nas conclusões estratégicas.

    **Dados da Análise:**
    {json.dumps(_analysis_results, ensure_ascii=False)}

    **Estruture sua resposta em Markdown, usando exatamente estes três tópicos:**

    ### Briefing Estratégico para Produto e Marketing

    **1. Diagnóstico Central:**
    * **Análise Profunda (2 parágrafos):** Considerando a análise de um grande volume de comentários, vá além da superfície. Sintetize os dados para revelar a história central.
        * **Qual é a principal força positiva?** Descreva o principal desejo ou motivação que atrai o público a este tópico/produto.
        * **Qual é a principal barreira ou fonte de atrito?** Descreva a maior dor, medo ou frustração que o público enfrenta.
        * **Qual a tensão estratégica resultante?** Explique o conflito central que surge do choque entre o desejo e a barreira. Esta tensão é a oportunidade de negócio mais importante a ser resolvida.

    **2. Diretrizes Estratégicas para MARKETING:**
    * Com base no diagnóstico, liste de 2 a 3 recomendações de alto impacto para a equipe de marketing. Seja prescritivo.

    **3. Diretrizes Estratégicas para PRODUTO:**
    * Com base no diagnóstico, liste de 2 a 3 recomendações de alto impacto para a equipe de produto. Seja prescritivo.

    O objetivo é criar um documento que um C-level ou líder de equipe possa ler em 60 segundos e entender exatamente onde focar os esforços.
    """
    try: response = model.generate_content(prompt); return response.text.strip()
    except Exception as e: return f"Erro ao gerar insights de PMM: {e}"

# --- FUNÇÕES DE VISUALIZAÇÃO ---
def plot_sentiment_chart(sentiment_data):
    colors_for_pie = {'positive': '#4CAF50', 'neutral': '#cccccc', 'negative': '#ffcd03', 'no_sentiment_detected': '#f0f0f0'}
    labels_order = ['positive', 'neutral', 'negative', 'no_sentiment_detected']
    display_labels = ['Positivo', 'Neutro', 'Negativo', 'Não Detectado']
    sizes = [sentiment_data.get(label, 0.0) for label in labels_order]
    filtered_data = [(display_labels[i], sizes[i], colors_for_pie[labels_order[i]]) for i, size in enumerate(sizes) if size > 0]
    if not filtered_data: st.warning("Dados de sentimento insuficientes."); return
    filtered_labels, filtered_sizes, filtered_colors = zip(*filtered_data)
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(filtered_sizes, explode=[0.03]*len(filtered_labels), labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    for autotext in autotexts: autotext.set_color('black'); autotext.set_fontweight('bold')
    ax.add_artist(plt.Circle((0,0),0.70,fc='white'))
    ax.axis('equal'); ax.set_title('1. Análise de Sentimento Geral', pad=18, color='#1f2329')
    st.pyplot(fig)

def plot_topics_chart(topics_data):
    if not topics_data: st.warning("Dados de temas insuficientes."); return
    df_topics = pd.DataFrame(topics_data).fillna(0)
    df_topics['Total'] = df_topics['positive'] + df_topics['neutral'] + df_topics['negative']
    df_topics = df_topics.sort_values('Total', ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_topics) * 0.5)))
    df_topics[['positive', 'neutral', 'negative']].plot(kind='barh', stacked=True, color=['#4CAF50', '#ffcd03', '#000000'], ax=ax)
    ax.set_yticklabels(df_topics['name'])
    ax.set_title('2. Temas Mais Citados por Sentimento'); ax.set_xlabel('Número de Comentários'); ax.set_ylabel('Tema')
    ax.legend(['Positivo', 'Neutro', 'Negativo'], loc='lower right', frameon=False)
    plt.tight_layout(); st.pyplot(fig)

def plot_word_cloud(term_clusters_data):
    if not term_clusters_data: st.warning("Dados de termos insuficientes."); return
    color_func = lambda *args, **kwargs: "#ffcd03" if np.random.rand() > 0.7 else "#1f2329"
    wordcloud = WordCloud(width=700, height=400, background_color='white', color_func=color_func, collocations=False).generate_from_frequencies(term_clusters_data)
    fig = plt.figure(figsize=(8, 5)); plt.imshow(wordcloud, interpolation='bilinear'); plt.axis('off'); plt.title('3. Agrupamento de Termos')
    st.pyplot(fig)

def plot_topic_relations_chart(topic_relations_data):
    if not topic_relations_data: st.warning("Dados de relações insuficientes."); return
    G = nx.Graph()
    for rel in topic_relations_data:
        if rel.get('source') and rel.get('target'): G.add_edge(rel['source'], rel['target'])
    if not G.edges(): st.warning("Nenhuma relação válida encontrada."); return
    fig, ax = plt.subplots(figsize=(8, 7)); pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#ffcd03', alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color='#1f2329', alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='#000000', ax=ax)
    ax.set_title('4. Relação Entre Temas'); plt.axis('off'); plt.tight_layout(); st.pyplot(fig)

# --- LAYOUT DA APLICAÇÃO STREAMLIT ---
st.set_page_config(layout="wide", page_title="Social Listening Tool + AI")
st.title("🗣️ Social Listening Tool + AI")
st.markdown("---")

st.markdown(f"Carregue uma base de comentários, insira até 3 URLs de vídeos do YouTube ou cole comentários abaixo. A análise da IA será feita com uma amostra de até **{MAX_COMMENTS_TO_PROCESS} comentários**.")

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
        for url in youtube_urls: temp_comments.extend(download_youtube_comments(url))
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
            st.subheader("Gráfico de Sentimento Geral")
            plot_sentiment_chart(analysis_results.get('sentiment', {}))
            st.subheader("Análise dos Insights do Gráfico")
            with st.spinner("Gerando análise..."):
                st.markdown(generate_sentiment_analysis_text(analysis_results.get('sentiment', {})))
        with tabs[1]:
            st.subheader("Gráfico de Temas por Sentimento")
            plot_topics_chart(analysis_results.get('topics', []))
            st.subheader("Análise Estratégica dos Temas")
            with st.spinner("Gerando análise..."):
                st.markdown(generate_topics_analysis_text(analysis_results.get('topics', [])))
        with tabs[2]:
            st.subheader("Nuvem de Termos-Chave")
            plot_word_cloud(analysis_results.get('term_clusters', {}))
            st.subheader("Análise dos Insights do Gráfico")
            with st.spinner("Gerando análise..."):
                st.markdown(generate_wordcloud_analysis_text(analysis_results.get('term_clusters', {})))
        with tabs[3]:
            st.subheader("Grafo de Relação Entre Temas")
            plot_topic_relations_chart(analysis_results.get('topic_relations', []))
            st.subheader("Análise dos Insights do Gráfico")
            with st.spinner("Gerando análise..."):
                st.markdown(generate_relations_analysis_text(analysis_results.get('topic_relations', [])))
        with tabs[4]:
            st.subheader("Análise Qualitativa Geral")
            with st.spinner("Gerando análise..."):
                st.markdown(generate_qualitative_analysis(analysis_results, text_to_analyze))
        with tabs[5]:
            st.subheader("Perfil da Persona Sintética")
            with st.spinner("Gerando persona..."):
                st.markdown(generate_persona_insights(analysis_results, text_to_analyze))
        with tabs[6]:
            st.subheader("Sugestões de Testes de Growth (ICE Score)")
            with st.spinner("Gerando testes..."):
                ice = generate_ice_score_tests(analysis_results)
                if ice: st.dataframe(pd.DataFrame(ice), hide_index=True, use_container_width=True)
                else: st.warning("Não foi possível gerar os testes de Growth.")
        with tabs[7]:
            st.subheader("Briefing Estratégico para Produto e Marketing")
            with st.spinner("Gerando briefing..."):
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
