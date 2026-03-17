graph TD
    A[Início] --> B[1. Inicialização do Sistema];
    B --> C{Definir Parâmetros: 𝑝𝑑𝑒𝑡, 𝛿𝑝, 𝛿𝑣, etc.};
    C --> D[Inicializar Vértices: 𝑉𝑖=0.05, 𝑜𝑤𝑛𝑒𝑟𝑖=∅];
    D --> E[Posicionar Partículas: Aleatório ou Ponderado por Grau];
    
    E --> F[Início da Iteração];
    
    F --> G{Para cada Partícula 𝑝𝑗};
    G --> H{Decisão de Movimento};
    H -- Prob. 𝑝𝑑𝑒𝑡 --> I[Determinístico: Move para vértice já visitado];
    H -- Prob. 1-𝑝𝑑𝑒𝑡 --> J[Exploratório: Move para vértice vizinho];
    
    I --> K[3. Atualização de Potenciais];
    J --> K;
    
    K --> L{Análise da Posse do Vértice Visitado 𝑣𝑖};
    L -- Vértice Livre --> M[Partícula assume posse<br>𝑉𝑖 = 𝑃𝑗];
    L -- Vértice Próprio --> N[Reforça Posse<br>𝑃𝑗 aumenta (Eq. 1)];
    L -- Vértice Oponente --> O[Disputa<br>𝑃𝑗 diminui (Eq. 2)<br>𝑉𝑖 diminui (Eq. 3)];

    M --> P{Verificar Limiar};
    N --> P;
    O --> P;
    
    P -- Potencial < 0.05 --> Q[Partícula é reinicializada ou<br>Vértice perde proprietário];
    P -- Potencial ≥ 0.05 --> R;
    Q --> R[Fim da análise da partícula];

    R --> S{4. Critérios de Convergência};
    S -- Não Satisfeito --> F;
    S -- Satisfeito --> T[Fim da Competição];
    
    T --> U[5. Pseudo-rotulagem];
    U --> V[Classificar Grupos (Positivo/Negativo)];
    V --> W[Calcular Dissimilaridade para Vértices em Grupos Negativos];
    W --> X[Ranquerar e Selecionar Top-k Vértices para o conjunto RN];
    
    X --> Z[Fim];