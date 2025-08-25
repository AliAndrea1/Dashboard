import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Profissional - CP1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Esconder elementos do Streamlit */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > div > div > div > div > div > section > div {
        padding-top: 1rem;
    }
    
    /* Título principal mais elegante */
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 2px;
        border-bottom: 3px solid #F3DCF3;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: white;
        border-left: 4px solid #F3DCF3;
        padding-left: 1rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #F3DCF3 0%, #A17FAA 100%);
        color: black;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Barras de habilidade com animação */
    .skill-bar {
        background-color: #000000;
        border-radius: 25px;
        padding: 4px;
        margin: 8px 0;
        overflow: hidden;
    }
    
    .skill-progress {
        background: linear-gradient(90deg, #F3DCF3, #A17FAA);
        height: 25px;
        border-radius: 20px;
        text-align: center;
        line-height: 25px;
        color: black;
        font-weight: 600;
        transition: width 2s ease-in-out;
    }
    
    .highlight-box {
        background: #F3DCF3;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #F3DCF3;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: black;
    }
    
    .success-box {
        background: linear-gradient(135deg, #F3DCF3 0%, #A17FAA 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #F3DCF3;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: black;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #F3DCF3 0%, #A17FAA 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #F3DCF3;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar personalizada */
    .css-1d391kg {
        background: linear-gradient(180deg, #F3DCF3 0%, #A17FAA 100%);
    }
    
    /* Métricas personalizadas */
    .metric-container {
        background: #F3DCF3;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
        color: #000000;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #000000;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #F3DCF3;
        margin-top: 0.5rem;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega os dados do Excel"""
    try:
        df = pd.read_excel('df_selecionado.xlsx')
        # Criar coluna de promoção
        df['Tem_Promocao'] = df['IDs_Promocao'].notna()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Função para calcular estatísticas descritivas
def calcular_estatisticas(serie):
    """Calcula estatísticas descritivas de uma série"""
    return {
        'count': len(serie),
        'mean': serie.mean(),
        'median': serie.median(),
        'mode': serie.mode().iloc[0] if len(serie.mode()) > 0 else np.nan,
        'std': serie.std(),
        'var': serie.var(),
        'min': serie.min(),
        'max': serie.max(),
        'q25': serie.quantile(0.25),
        'q75': serie.quantile(0.75),
        'iqr': serie.quantile(0.75) - serie.quantile(0.25),
        'cv': (serie.std() / serie.mean()) * 100
    }

# Função para calcular intervalo de confiança
def calcular_ic(dados, confianca=0.95):
    """Calcula intervalo de confiança para a média"""
    n = len(dados)
    media = dados.mean()
    desvio = dados.std(ddof=1)
    alpha = 1 - confianca
    t_critico = stats.t.ppf(1 - alpha/2, n-1)
    erro_padrao = desvio / np.sqrt(n)
    margem_erro = t_critico * erro_padrao
    
    return {
        'media': media,
        'ic_inferior': media - margem_erro,
        'ic_superior': media + margem_erro,
        'margem_erro': margem_erro,
        'erro_padrao': erro_padrao,
        't_critico': t_critico
    }

# Função para teste t
def teste_t_independente(grupo1, grupo2, teste_unilateral=True):
    """Realiza teste t para amostras independentes"""
    # Teste de Levene para igualdade de variâncias
    levene_stat, levene_p = stats.levene(grupo1, grupo2)
    equal_var = levene_p > 0.05
    
    # Teste t
    t_stat, p_bilateral = stats.ttest_ind(grupo1, grupo2, equal_var=equal_var)
    
    # P-valor unilateral se necessário
    if teste_unilateral:
        p_unilateral = p_bilateral / 2 if t_stat > 0 else 1 - (p_bilateral / 2)
    else:
        p_unilateral = p_bilateral
    
    return {
        't_stat': t_stat,
        'p_bilateral': p_bilateral,
        'p_unilateral': p_unilateral,
        'levene_p': levene_p,
        'equal_var': equal_var
    }

# Sidebar para navegação - Design mais limpo
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; color: white;'>
    <h2 style='color: #F3DCF3; margin-bottom: 2rem;'>Dashboard</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navegação:",
    ["🏠 Início", "🎓 Perfil Profissional", "💻 Competências", "📈 Análise de Dados"],
    index=0
)

# Página Home - Design mais moderno
if page == "🏠 Início":
    # Título principal mais elegante (sem opção de aumentar)
    st.markdown('<div class="main-header">Dashboard Profissional</div>', unsafe_allow_html=True)
    
    # Layout mais limpo para a foto
    col1, col2, col3 = st.columns([1, 1, 1])
    
    st.markdown('<div class="section-header">Sobre Mim</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
    <p style='font-size: 1.1rem; line-height: 1.6;'>
    Sou estudante de Engenharia de Software na FIAP, formada em Desenvolvimento de Sistemas pela ETEC Professor Horácio Augusto da Silveira. 
    Tenho inglês intermediário e espanhol avançado, e já participei de projetos acadêmicos e práticos que me permitiram desenvolver tanto minha 
    capacidade técnica quanto criativa.
    </p>
    <p style='font-size: 1.1rem; line-height: 1.6; '>
    <strong>Objetivo:</strong> Em busca da primeira oportunidade profissional na área de tecnologia, com foco em análise de dados,
automação e desenvolvimento de soluções com Python e SQL.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Principais Funcionalidades</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin-top: 0;'>📊 Análise Descritiva</h3>
            <p>Medidas de tendência central, dispersão e visualizações interativas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin-top: 0;'>🎯 Testes de Hipótese</h3>
            <p>Comparações estatísticas entre grupos com interpretação de resultados</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style='margin-top: 0;'>📈 Intervalos de Confiança</h3>
            <p>Estimação de parâmetros populacionais com nível de confiança</p>
        </div>
        """, unsafe_allow_html=True)

# Página Perfil Profissional - Renomeada e redesenhada
elif page == "🎓 Perfil Profissional":
    st.markdown('<div class="main-header">Perfil Profissional</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Formação Acadêmica</div>', unsafe_allow_html=True)
    
    with st.expander("🎓 Formação em Desenvolvimento de Sistemas", expanded=True):
        st.markdown("""
        <div class="highlight-box">
    <p><strong>Instituição:</strong> Escola Técnica Estadual Professor Horácio Augusto da Silveira</p>
    <p><strong>Curso:</strong> Técnico em Desenvolvimento de Sistemas</p>
    <p><strong>Período:</strong> 2021 - 2023</p>
    <p><strong>Principais Disciplinas:</strong></p>
    <ul>
        <li>Estatística Aplicada e Probabilidade</li>
        <li>Programação em Python para Análise de Dados</li>
        <li>Banco de Dados e SQL</li>
        <li>Visualização de Dados e Business Intelligence</li>
        <li>Programação Front-end (HTML, CSS, JavaScript)</li>
        <li>Análise e Projeto de Sistemas</li>
    </ul>
</div>
        """, unsafe_allow_html=True)
    
    with st.expander("📚 Certificações e Cursos Complementares"):
        st.markdown("""
      <div class="success-box">
    <ul>
        <li><strong>Qualificação Profissional Técnica de Nível Médio de Auxiliar em Desenvolvimento de Sistemas</strong> - ETEC (2023)</li>
        <li><strong>Qualificação Profissional Técnica de Nível Médio de Programador de Computadores</strong> - ETEC (2023)</li>
        <li><strong>Ensino Médio com Habilitação Profissional de Técnico em Desenvolvimento de Sistemas</strong> - ETEC (2023)</li>
        <li><strong>Formação Social e Sustentabilidade</strong> - FIAP (2024)</li>
        <li><strong>Design Thinking - Process</strong> - FIAP (2024)</li>
        <li><strong>Gestão de Infraestrutura de TI</strong> - FIAP (2024)</li>
        <li><strong>Lógica de Programação</strong> - Alura (2024)</li>
        <li><strong>Inglês (nível intermediário)</strong> - CCAA (2024)</li>
    </ul>
</div>

        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Experiência e Projetos</div>', unsafe_allow_html=True)
    
    with st.expander("💼 Projetos Acadêmicos Relevantes", expanded=True):
        st.markdown("""
       <div class="highlight-box">
    <h4>🏎️ SustenRace - Fórmula E (2024)</h4>
    <p><strong>Descrição:</strong> Desenvolvimento de plataforma interativa para aumentar a popularidade da Fórmula E e engajar fãs com carros sustentáveis</p>
    <p><strong>Tecnologias:</strong> HTML, CSS, JavaScript, React, Three.js</p>
    <p><strong>Principais Realizações:</strong></p>
    <ul>
        <li>Criação de visualização 3D interativa de carros da Fórmula E</li>
        <li>Implementação de funcionalidades para explorar peças internas e informações do veículo</li>
        <li>Desenvolvimento de mecânicas de interação para engajamento do público</li>
        <li>Planejamento de narrativa digital destacando sustentabilidade e inovação tecnológica</li>
    </ul>
</div>
                    <div class="highlight-box">
    <h4>🍷 Projeto de Transformação Digital em Vinharia (2024)</h4>
    <p><strong>Descrição:</strong> Desenvolvimento de solução para adequar a vinharia ao modelo de entrega online durante a pandemia, com foco em digitalização de processos e melhoria da experiência do cliente</p>
    <p><strong>Tecnologias:</strong> HTML, CSS, JavaScript, Python, Excel</p>
    <p><strong>Principais Realizações:</strong></p>
    <ul>
        <li>Mapeamento de processos de venda e logística da vinharia</li>
        <li>Criação de sistema de pedidos online integrado ao estoque</li>
        <li>Automatização de relatórios de vendas e controle de inventário</li>
        <li>Melhoria da experiência do cliente por meio de interface web intuitiva</li>
    </ul>
</div>
        """, unsafe_allow_html=True)

# Página Competências - Redesenhada
elif page == "💻 Competências":
    st.markdown('<div class="main-header">Competências Técnicas</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Linguagens de Programação</div>', unsafe_allow_html=True)
        
        skills_prog = {
            "Python": 90,
            "SQL": 85,
            "Java": 70,
            "JavaScript": 60
        }
        
        for skill, level in skills_prog.items():
            st.markdown(f"""
            <div class="skill-bar">
                <div class="skill-progress" style="width: {level}%;">
                    {skill} - {level}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Ferramentas e Tecnologias</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
    <h4>📊 Análise de Dados</h4>
    <ul>
        <li>Excel Básico</li>
        <li>Power BI</li>
        <li>Jupyter Notebook</li>
        <li>Python</li>
        <li>SQL</li>
    </ul>
</div>

        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
        <h4>🗄️ Banco de Dados</h4>
        <ul>
            <li>MySQL</li>
            <li>PostgreSQL</li>
            <li>SQLite</li>
            <li>MongoDB (básico)</li>
            <li>BigQuery</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="highlight-box">
        <h4>⚙️ Desenvolvimento</h4>
        <ul>
            <li>Git/GitHub</li>
            <li>VS Code</li>
            <li>Docker (básico)</li>
            <li>Linux/Ubuntu</li>
            <li>APIs REST</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Competências Comportamentais</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>🧠 Habilidades Analíticas</h4>
        <ul>
            <li><strong>Pensamento Crítico:</strong> Capacidade de questionar dados e resultados</li>
            <li><strong>Resolução de Problemas:</strong> Abordagem sistemática para desafios complexos</li>
            <li><strong>Atenção aos Detalhes:</strong> Precisão na análise e interpretação</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
      <div class="success-box">
    <h4>🤝 Habilidades Interpessoais</h4>
    <ul>
        <li><strong>Comunicação:</strong> Apresentação clara de insights técnicos</li>
        <li><strong>Liderança:</strong> Coordenação e organização de projetos acadêmicos e práticos</li>
        <li><strong>Aprendizado Contínuo:</strong> Adaptação a novas tecnologias</li>
    </ul>
</div>

        """, unsafe_allow_html=True)

# Página Análise de Dados - Mantida com melhorias visuais
# ... (código anterior permanece igual)

elif page == "📈 Análise de Dados":
    st.markdown('<div class="main-header">Análise de Dados: Vendas E-commerce</div>', unsafe_allow_html=True)
    
    # Carregar dados
    df = load_data()
    
    if df is not None:
        # 1. APRESENTAÇÃO DOS DADOS
        st.markdown('<div class="section-header">1. Apresentação dos Dados e Tipos de Variáveis</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total de Registros</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Total de Colunas</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">Mar-Jun</div>
                <div class="metric-label">Período (2022)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">R$ {df['Valor_Pedido_BRL'].sum():,.0f}</div>
                <div class="metric-label">Valor Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
        <h4>📋 Descrição do Dataset</h4>
        <p>Este dataset contém informações detalhadas sobre vendas de um e-commerce de roupas, 
        cobrindo o período de Abril a junho de 2022.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dicionário para personalizar as descrições das variáveis
        informacoes_colunas = {
            "Index": "Inteiro, provavelmente um índice sequencial",
            "Qty": "Inteir, quantidade de itens no pedido",
            "Valor_Pedido": "Float, valor do pedido",
            "CEP_Destino": "Float, CEP de destino (pode ser tratado como categórico se não for usado para cálculos numéricos)",
            "Valor_Pedido_BRL": "Float, valor do pedido em BRL",
            "Data_Pedido": "Data, data do pedido",
            "Categoria": "Texto, categoria do produto",
            "Status_Pedido": "Texto, status atual do pedido",
            "Venda_B2B": "Booleano, indica se é venda B2B",
            "IDs_Promocao": "Texto, IDs das promoções aplicadas",
            "Tem_Promocao": "Booleano, indica se tem promoção"
        }
        
        # Mostrar tipos de variáveis com descrições personalizadas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
            <h4>🔢 Variáveis Numéricas</h4>
            """, unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                descricao = informacoes_colunas.get(col, f"{df[col].dtype} - Sem descrição personalizada")
                st.write(f"• **{col}**: {descricao}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
            <h4>📝 Variáveis Categóricas</h4>
            """, unsafe_allow_html=True)
            categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
            for col in categorical_cols[:10]:
                descricao = informacoes_colunas.get(col, f"{df[col].dtype} - Sem descrição personalizada")
                st.write(f"• **{col}**: {descricao}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Perguntas de análise
        st.markdown("""
        <div class="highlight-box">
        <h4>🎯 Perguntas de Análise</h4>
        <ol>
        <li><strong>Quais são os produtos mais vendidos e quais geram mais receita?</strong></li>
        <li><strong>Como estão distribuídos os valores dos pedidos?</strong></li>
        <li><strong>Qual a porcentagem de status dos pedidos?</strong></li>
        <li><strong>Os pedidos B2B possuem valores médios maiores do que os pedidos B2C?</strong> (Teste de Hipótese)</li>
        <li><strong>Qual o valor médio real de um pedido nesta plataforma?</strong> (Intervalo de Confiança)</li>
        <li><strong>Promoções realmente aumentam o valor médio dos pedidos?</strong> (Teste de Hipótese)</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. ANÁLISE DESCRITIVA
        st.markdown('<div class="section-header">2. Medidas Centrais, Dispersão e Análise Inicial</div>', unsafe_allow_html=True)
        
        # Análise da variável principal: Valor_Pedido_BRL
        valores_pedidos = df['Valor_Pedido_BRL'].dropna()
        stats_pedidos = calcular_estatisticas(valores_pedidos)
        
        st.write("### 📊 Análise da Variável Principal: Valor dos Pedidos (R$)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>📈 Medidas de Tendência Central</h4>
            """, unsafe_allow_html=True)
            st.metric("Média", f"R$ {stats_pedidos['mean']:.2f}")
            st.metric("Mediana", f"R$ {stats_pedidos['median']:.2f}")
            st.metric("Moda", f"R$ {stats_pedidos['mode']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>📊 Medidas de Dispersão</h4>
            """, unsafe_allow_html=True)
            st.metric("Desvio Padrão", f"R$ {stats_pedidos['std']:.2f}")
            st.metric("Variância", f"R$ {stats_pedidos['var']:.2f}")
            st.metric("Coef. Variação", f"{stats_pedidos['cv']:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="highlight-box">
            <h4>📋 Quartis</h4>
            """, unsafe_allow_html=True)
            st.metric("Q1 (25%)", f"R$ {stats_pedidos['q25']:.2f}")
            st.metric("Q3 (75%)", f"R$ {stats_pedidos['q75']:.2f}")
            st.metric("IQR", f"R$ {stats_pedidos['iqr']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Interpretação das medidas
        st.markdown("""
        <div class="highlight-box">
        <h4>📈 Interpretação das Medidas Estatísticas</h4>
        <p><strong>Distribuição:</strong> A média (R$ {:.2f}) é ligeiramente superior à mediana (R$ {:.2f}), 
        indicando uma distribuição com assimetria positiva (cauda à direita).</p>
        <p><strong>Variabilidade:</strong> O coeficiente de variação de {:.1f}% indica uma dispersão moderada 
        nos valores dos pedidos.</p>
        <p><strong>Concentração:</strong> 50% dos pedidos estão entre R$ {:.2f} e R$ {:.2f} (IQR = R$ {:.2f}).</p>
        </div>
        """.format(stats_pedidos['mean'], stats_pedidos['median'], stats_pedidos['cv'], 
                  stats_pedidos['q25'], stats_pedidos['q75'], stats_pedidos['iqr']), 
        unsafe_allow_html=True)
        
        # Visualizações da distribuição
        st.write("### 📊 Visualizações da Distribuição dos Valores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma
            fig_hist = px.histogram(
                valores_pedidos, 
                nbins=50,
                title="Distribuição dos Valores dos Pedidos",
                labels={'value': 'Valor do Pedido (R$)', 'count': 'Frequência'},
                color_discrete_sequence=['#3498db']
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Boxplot
            fig_box = px.box(
                y=valores_pedidos,
                title="Boxplot dos Valores dos Pedidos",
                labels={'y': 'Valor do Pedido (R$)'},
                color_discrete_sequence=['#e74c3c']
            )
            fig_box.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Análise de produtos mais vendidos
        st.write("### 🛍️ Produtos Mais Vendidos e Receita por Categoria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Produtos mais vendidos
            produtos_vendidos = df['Categoria'].value_counts().head(10)
            fig_vendidos = px.bar(
                x=produtos_vendidos.values,
                y=produtos_vendidos.index,
                orientation='h',
                title="Top 10 Produtos Mais Vendidos",
                labels={'x': 'Quantidade de Vendas', 'y': 'Categoria'},
                color_discrete_sequence=['#2ecc71']
            )
            fig_vendidos.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_vendidos, use_container_width=True)
        
        with col2:
            # Receita por categoria
            receita_categoria = df.groupby('Categoria')['Valor_Pedido_BRL'].sum().sort_values(ascending=False).head(10)
            fig_receita = px.bar(
                x=receita_categoria.values,
                y=receita_categoria.index,
                orientation='h',
                title="Top 10 Categorias por Receita",
                labels={'x': 'Receita Total (R$)', 'y': 'Categoria'},
                color_discrete_sequence=['#9b59b6']
            )
            fig_receita.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_receita, use_container_width=True)
        
        # Status dos pedidos
        st.write("### 📦 Status dos Pedidos")
        
        status_pedidos = df['Status_Pedido'].value_counts()
        fig_status = px.pie(
            values=status_pedidos.values,
            names=status_pedidos.index,
            title="Distribuição dos Status dos Pedidos",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_status.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_status, use_container_width=True)
        
        # 3. INTERVALOS DE CONFIANÇA E TESTES DE HIPÓTESE
        st.markdown('<div class="section-header">3. Intervalos de Confiança e Testes de Hipótese</div>', unsafe_allow_html=True)
        
        # 3.1 Intervalo de Confiança
        st.write("### 🎯 Intervalo de Confiança para a Média dos Pedidos")
        
        ic_resultado = calcular_ic(valores_pedidos, 0.95)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="success-box">
            <h4>📊 Intervalo de Confiança (95%)</h4>
            <p><strong>Média amostral:</strong> R$ {ic_resultado['media']:.2f}</p>
            <p><strong>Intervalo:</strong> [R$ {ic_resultado['ic_inferior']:.2f} ; R$ {ic_resultado['ic_superior']:.2f}]</p>
            <p><strong>Margem de erro:</strong> R$ {ic_resultado['margem_erro']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Interpretação:**")
            st.write(f"Com 95% de confiança, o valor médio real dos pedidos na plataforma está entre R$ {ic_resultado['ic_inferior']:.2f} e R$ {ic_resultado['ic_superior']:.2f}")
        
        with col2:
            # Visualização do IC
            fig_ic = go.Figure()
            
            # Adicionar a distribuição normal
            x_range = np.linspace(ic_resultado['ic_inferior'] - 5, ic_resultado['ic_superior'] + 5, 100)
            y_normal = stats.norm.pdf(x_range, ic_resultado['media'], ic_resultado['erro_padrao'])
            
            fig_ic.add_trace(go.Scatter(x=x_range, y=y_normal, mode='lines', name='Distribuição da Média', line=dict(color='#3498db')))
            fig_ic.add_vline(x=ic_resultado['media'], line_dash="solid", annotation_text="Média", line=dict(color='#e74c3c'))
            fig_ic.add_vline(x=ic_resultado['ic_inferior'], line_dash="dash", annotation_text="IC Inferior", line=dict(color='#f39c12'))
            fig_ic.add_vline(x=ic_resultado['ic_superior'], line_dash="dash", annotation_text="IC Superior", line=dict(color='#f39c12'))
            
            fig_ic.update_layout(
                title="Intervalo de Confiança (95%)", 
                xaxis_title="Valor (R$)", 
                yaxis_title="Densidade",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_ic, use_container_width=True)
        
        # 3.2 Teste de Hipótese: B2B vs B2C
        st.write("### 🏢 Teste de Hipótese: B2B vs B2C")
        
        # Separar dados
        b2c_valores = df[df['Venda_B2B'] == False]['Valor_Pedido_BRL'].dropna()
        b2b_valores = df[df['Venda_B2B'] == True]['Valor_Pedido_BRL'].dropna()
        
        # Realizar teste
        teste_b2b = teste_t_independente(b2b_valores, b2c_valores, teste_unilateral=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
            <h4>📊 Estatísticas Descritivas</h4>
            """, unsafe_allow_html=True)
            st.write(f"• B2C: {len(b2c_valores):,} pedidos, média R$ {b2c_valores.mean():.2f}")
            st.write(f"• B2B: {len(b2b_valores):,} pedidos, média R$ {b2b_valores.mean():.2f}")
            
            st.write("**Hipóteses:**")
            st.write("• H₀: μ_B2B = μ_B2C (não há diferença)")
            st.write("• H₁: μ_B2B > μ_B2C (B2B tem média maior)")
            st.write("• Nível de significância: α = 0.05")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if teste_b2b['p_unilateral'] < 0.05:
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Resultado: SIGNIFICATIVO</h4>
                <p><strong>Estatística t:</strong> {teste_b2b['t_stat']:.4f}</p>
                <p><strong>P-valor:</strong> {teste_b2b['p_unilateral']:.4f}</p>
                <p><strong>Conclusão:</strong> Rejeitamos H₀. Há evidência estatística de que pedidos B2B têm valor médio maior que B2C.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                <h4>❌ Resultado: NÃO SIGNIFICATIVO</h4>
                <p><strong>Estatística t:</strong> {teste_b2b['t_stat']:.4f}</p>
                <p><strong>P-valor:</strong> {teste_b2b['p_unilateral']:.4f}</p>
                <p><strong>Conclusão:</strong> Não rejeitamos H₀. Não há evidência suficiente de diferença.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Boxplot comparativo
            df_plot = df[['Venda_B2B', 'Valor_Pedido_BRL']].dropna()
            fig_b2b = px.box(
                df_plot, 
                x='Venda_B2B', 
                y='Valor_Pedido_BRL',
                title="Comparação B2B vs B2C",
                labels={'Venda_B2B': 'Tipo de Venda', 'Valor_Pedido_BRL': 'Valor do Pedido (R$)'},
                color='Venda_B2B',
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            fig_b2b.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_b2b, use_container_width=True)
        
        # 3.3 Teste de Hipótese: Promoções
        st.write("### 🎁 Teste de Hipótese: Promoções vs Sem Promoções")
        
        # Separar dados
        sem_promo = df[df['Tem_Promocao'] == False]['Valor_Pedido_BRL'].dropna()
        com_promo = df[df['Tem_Promocao'] == True]['Valor_Pedido_BRL'].dropna()
        
        # Realizar teste
        teste_promo = teste_t_independente(com_promo, sem_promo, teste_unilateral=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
            <h4>📊 Estatísticas Descritivas</h4>
            """, unsafe_allow_html=True)
            st.write(f"• Sem promoção: {len(sem_promo):,} pedidos, média R$ {sem_promo.mean():.2f}")
            st.write(f"• Com promoção: {len(com_promo):,} pedidos, média R$ {com_promo.mean():.2f}")
            
            st.write("**Hipóteses:**")
            st.write("• H₀: μ_com_promo = μ_sem_promo (não há diferença)")
            st.write("• H₁: μ_com_promo > μ_sem_promo (promoções aumentam valor)")
            st.write("• Nível de significância: α = 0.05")
            st.markdown("</div>", unsafe_allow_html=True)
            
            if teste_promo['p_unilateral'] < 0.05:
                st.markdown(f"""
                <div class="success-box">
                <h4>✅ Resultado: SIGNIFICATIVO</h4>
                <p><strong>Estatística t:</strong> {teste_promo['t_stat']:.4f}</p>
                <p><strong>P-valor:</strong> {teste_promo['p_unilateral']:.4f}</p>
                <p><strong>Conclusão:</strong> Rejeitamos H₀. Há evidência estatística de que promoções aumentam o valor médio dos pedidos.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                <h4>❌ Resultado: NÃO SIGNIFICATIVO</h4>
                <p><strong>Estatística t:</strong> {teste_promo['t_stat']:.4f}</p>
                <p><strong>P-valor:</strong> {teste_promo['p_unilateral']:.4f}</p>
                <p><strong>Conclusão:</strong> Não rejeitamos H₀. Não há evidência suficiente de diferença.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Boxplot comparativo
            df_plot2 = df[['Tem_Promocao', 'Valor_Pedido_BRL']].dropna()
            fig_promo = px.box(
                df_plot2, 
                x='Tem_Promocao', 
                y='Valor_Pedido_BRL',
                title="Comparação: Com vs Sem Promoção",
                labels={'Tem_Promocao': 'Tem Promoção', 'Valor_Pedido_BRL': 'Valor do Pedido (R$)'},
                color='Tem_Promocao',
                color_discrete_sequence=['#f39c12', '#27ae60']
            )
            fig_promo.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_promo, use_container_width=True)
        
        # Resumo Final
        st.markdown('<div class="section-header">4. Resumo dos Resultados Estatísticos</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="highlight-box">
        <h4>📋 Resumo Executivo</h4>
        
        <h5>1. Intervalo de Confiança (95%):</h5>
        <p>O valor médio real dos pedidos está entre <strong>R$ {ic_resultado['ic_inferior']:.2f}</strong> e <strong>R$ {ic_resultado['ic_superior']:.2f}</strong></p>
        
        <h5>2. Teste B2B vs B2C:</h5>
        <p>{'✅ SIGNIFICATIVO' if teste_b2b['p_unilateral'] < 0.05 else '❌ NÃO SIGNIFICATIVO'} (p = {teste_b2b['p_unilateral']:.4f})</p>
        <p>{'Pedidos B2B têm valor médio significativamente maior que B2C' if teste_b2b['p_unilateral'] < 0.05 else 'Não há diferença significativa entre B2B e B2C'}</p>
        
        <h5>3. Teste Promoções:</h5>
        <p>{'✅ SIGNIFICATIVO' if teste_promo['p_unilateral'] < 0.05 else '❌ NÃO SIGNIFICATIVO'} (p = {teste_promo['p_unilateral']:.4f})</p>
        <p>{'Promoções aumentam significativamente o valor médio dos pedidos' if teste_promo['p_unilateral'] < 0.05 else 'Não há evidência de que promoções aumentem o valor médio'}</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.error("Não foi possível carregar os dados. Verifique se o arquivo está no local correto.")

# Footer mais elegante
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>Dashboard desenvolvido para CP1 - Análise de Dados | 2025</p>
    <p style='font-size: 0.9rem;'>Demonstração completa de análise estatística aplicada a dados reais de e-commerce</p>
</div>
""", unsafe_allow_html=True)