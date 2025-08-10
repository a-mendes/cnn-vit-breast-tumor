# Modelo CNN-ViT Híbrido para Classificação de Tumores Mamários

## Visão Geral

Este documento apresenta uma explicação detalhada do modelo **CNNViTHybrid** implementado para classificação de tumores mamários em imagens de ultrassom. O modelo combina as vantagens das Redes Neurais Convolucionais (CNN) com os Transformers (Vision Transformer - ViT), criando uma arquitetura híbrida eficiente para diagnóstico médico.

## Dataset Utilizado

- **Nome**: BUSI (Breast Ultrasound Images Dataset)
- **Classes**: 3 categorias
  - `malignant`: Tumores malignos
  - `benign`: Tumores benignos  
  - `normal`: Tecido normal
- **Formato**: Imagens em escala de cinza com máscaras correspondentes
- **Divisão dos dados**:
  - Treino: 70%
  - Validação: 15%
  - Teste: 15%

## Arquitetura do Modelo CNN-ViT

### 1. CNN Backbone (Extrator de Características)

```python
if cnn_backbone == 'resnet50':
    cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    self.feature_extractor = nn.Sequential(*list(cnn.children())[:-2])
    cnn_output_channels = 2048
```

**Características:**
- Utiliza **ResNet-50** pré-treinado no ImageNet
- Remove as últimas duas camadas (pooling global e classificador)
- Mantém apenas as camadas convolucionais para extração de features espaciais
- **Entrada**: Imagem (3, 224, 224)
- **Saída**: Tensor de forma `(batch, 2048, 7, 7)`

**Vantagens:**
- Aproveitamento de pesos pré-treinados (transfer learning)
- Extração eficiente de características locais
- Redução de parâmetros treináveis

### 2. Projeção Dimensional

```python
self.projection = nn.Conv2d(cnn_output_channels, embed_dim, kernel_size=1)
```

**Função:**
- Converte features de 2048 canais para 256 canais (embedding dimension)
- Usa convolução 1×1 para redução dimensional eficiente
- **Entrada**: `(batch, 2048, 7, 7)`
- **Saída**: `(batch, 256, 7, 7)`

### 3. Preparação para o Transformer

#### Flattening e Reshaping
```python
b, c, h, w = x.shape
x = x.flatten(2).permute(0, 2, 1)  # Shape: (batch, 49, 256)
```

**Processo:**
- Converte grade 2D (7×7) em sequência de 49 patches
- Cada patch representa uma região espacial da imagem
- Transforma dados espaciais em formato sequencial para o Transformer

#### CLS Token
```python
cls_tokens = self.cls_token.expand(b, -1, -1)
x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch, 50, 256)
```

**Características:**
- Token especial adicionado no início da sequência
- Função: agregar informações de todos os patches para classificação final
- Inspirado na arquitetura ViT original
- **Resultado**: Sequência com 50 tokens (1 CLS + 49 patches)

#### Positional Encoding
```python
self.pos_encoder = nn.Parameter(torch.randn(1, 50, embed_dim))
x += self.pos_encoder
```

**Necessidade:**
- Transformers não têm noção inerente de posição
- Adiciona informação espacial a cada token
- Permite ao modelo distinguir entre patches de diferentes regiões

## 4. TransformerEncoderLayer e TransformerEncoder ⭐

### TransformerEncoderLayer

```python
encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
```

**Parâmetros principais:**
- `d_model=256`: Dimensão do embedding
- `nhead=8`: Número de cabeças de atenção
- `batch_first=True`: Formato de entrada (batch, seq_len, embed_dim)

#### Componentes do TransformerEncoderLayer:

##### 1. Multi-Head Self-Attention
**Funcionamento:**
- Cada token (patch) "observa" todos os outros tokens
- 8 cabeças de atenção capturam diferentes tipos de relações
- Calcula pesos de atenção entre todos os pares de tokens
- Permite identificação de regiões importantes para classificação

**Matemática:**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
MultiHead = Concat(head_1, ..., head_h)W^O
```

**Vantagens:**
- **Atenção Global**: cada região pode "ver" qualquer outra região
- **Múltiplas perspectivas**: diferentes cabeças capturam relações distintas
- **Paralelização**: processamento simultâneo de todos os tokens

##### 2. Feed-Forward Network (FFN)
```python
# Estrutura interna (simplificada)
ffn = nn.Sequential(
    nn.Linear(embed_dim, 4 * embed_dim),
    nn.ReLU(),
    nn.Linear(4 * embed_dim, embed_dim)
)
```

**Características:**
- Rede feedforward aplicada independentemente a cada token
- Expansão e contração dimensional (256 → 1024 → 256)
- Transformação não-linear das representações

##### 3. Residual Connections e Layer Normalization
```python
# Estrutura conceitual
output = LayerNorm(input + MultiHeadAttention(input))
output = LayerNorm(output + FFN(output))
```

**Benefícios:**
- **Skip connections**: facilitam o fluxo de gradientes
- **Layer normalization**: estabiliza o treinamento
- **Redes mais profundas**: permite empilhamento de múltiplas camadas

### TransformerEncoder

```python
self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

**Configuração:**
- `num_layers=6`: 6 camadas de encoder empilhadas
- Cada camada refina progressivamente as representações

#### Funcionamento em Camadas:

1. **Camada 1**: Processa sequência inicial (CLS + 49 patches)
   - Estabelece relações básicas entre patches
   - Identifica características locais importantes

2. **Camadas 2-3**: Refinamento intermediário
   - Combina informações de patches adjacentes
   - Detecta padrões de média escala

3. **Camadas 4-5**: Contexto de alto nível
   - Integra informações de toda a imagem
   - Identifica padrões complexos e relações globais

4. **Camada 6**: Representação final
   - Consolidação de todas as informações
   - Preparação para classificação

#### Vantagens do Transformer:

**1. Atenção Global**
- Cada pixel pode influenciar qualquer outro pixel
- Captura dependências de longo alcance
- Útil para detectar padrões distribuídos na imagem

**2. Paralelização**
- Processamento simultâneo de todos os tokens
- Eficiência computacional em GPUs
- Redução do tempo de treinamento

**3. Flexibilidade**
- Adapta-se a diferentes tamanhos de entrada
- Escalável para diferentes resoluções
- Transferível para outras tarefas

### 5. Classificação Final

```python
cls_output = x[:, 0]  # Extrai apenas o CLS token
output = self.classifier(cls_output)
```

**Processo:**
- Utiliza apenas a saída do CLS token
- O CLS token agregou informações de toda a imagem através das camadas de atenção
- Classificador linear mapeia para as 3 classes

## Fluxo Completo de Dados

### Pipeline de Processamento:

1. **Entrada**: Imagem ultrassom (1, 224, 224) → convertida para (3, 224, 224)
2. **CNN Features**: ResNet-50 → (batch, 2048, 7, 7)
3. **Projeção**: Conv1x1 → (batch, 256, 7, 7)
4. **Sequenciação**: Flatten → (batch, 49, 256)
5. **+ CLS Token**: Concatenação → (batch, 50, 256)
6. **+ Pos Encoding**: Adição → (batch, 50, 256)
7. **Transformer**: 6 camadas → (batch, 50, 256) contextualizadas
8. **Classificação**: CLS token → (batch, 3 classes)

### Transformações de Dados:

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Grayscale → RGB
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
```

## Configuração de Treinamento

### Hiperparâmetros:

```python
# Arquitetura
embed_dim = 256
nhead = 8
num_layers = 6

# Treinamento
num_epochs = 20
batch_size = 16
learning_rate = 0.0001
optimizer = AdamW
criterion = CrossEntropyLoss
```

### Processo de Treinamento:

1. **Fase de Treinamento**:
   - Forward pass através da arquitetura completa
   - Cálculo da loss usando CrossEntropyLoss
   - Backpropagation e atualização de pesos com AdamW

2. **Fase de Validação**:
   - Avaliação sem atualização de pesos
   - Monitoramento da performance de generalização
   - Salvamento do melhor modelo baseado na acurácia de validação

## Vantagens da Arquitetura CNN-ViT

### 1. **Eficiência Computacional**
- CNN extrai features locais eficientemente
- Transformer processa apenas 49 patches ao invés de 50,176 pixels
- Redução significativa da complexidade computacional

### 2. **Contexto Global**
- Transformer captura relações de longo alcance
- Útil para detectar padrões distribuídos em tumores
- Atenção entre regiões distantes da imagem

### 3. **Transfer Learning**
- Aproveitamento de pesos pré-treinados do ResNet-50
- Conhecimento prévio sobre características visuais gerais
- Convergência mais rápida e melhor performance

### 4. **Flexibilidade**
- Combina o melhor dos dois mundos (CNN + Transformer)
- Adaptável para diferentes tipos de imagens médicas
- Escalável para diferentes resoluções

### 5. **Interpretabilidade**
- Mapas de atenção mostram regiões importantes
- CLS token agrega informações interpretáveis
- Útil para análise médica e diagnóstico

## Aplicação em Imagens Médicas

### Características Relevantes:

1. **Detalhes Locais** (CNN):
   - Texturas específicas de tumores
   - Bordas e contornos
   - Padrões de densidade

2. **Padrões Globais** (Transformer):
   - Distribuição espacial de anomalias
   - Relações entre diferentes regiões
   - Contexto anatômico geral

### Vantagens para Diagnóstico:

- **Precisão**: Combinação de análise local e global
- **Robustez**: Menos sensível a variações locais
- **Generalização**: Melhor performance em dados não vistos
- **Interpretabilidade**: Visualização das regiões de atenção

## Avaliação e Métricas

### Métricas Utilizadas:
- **Acurácia**: Percentage de classificações corretas
- **Precision**: TP / (TP + FP) por classe
- **Recall**: TP / (TP + FN) por classe
- **F1-Score**: Média harmônica entre precision e recall
- **Matriz de Confusão**: Visualização detalhada das classificações

### Interpretação dos Resultados:
- Classification Report para análise detalhada por classe
- Confusion Matrix para identificar padrões de erro
- Visualização de predições em amostras de teste

## Possíveis Melhorias

### 1. **Arquitetura**:
- Diferentes backbones CNN (EfficientNet, DenseNet)
- Variação no número de camadas Transformer
- Diferentes estratégias de positional encoding

### 2. **Treinamento**:
- Hyperparameter tuning sistemático
- Diferentes optimizers e schedulers
- Técnicas de regularização avançadas

### 3. **Dados**:
- Aumento de dados mais agressivo
- Balanceamento de classes
- Uso de máscaras para atenção dirigida

### 4. **Avaliação**:
- Validação cruzada
- Métricas específicas para diagnóstico médico
- Análise de interpretabilidade com mapas de atenção

## Conclusão

O modelo CNNViTHybrid representa uma abordagem inovadora para classificação de tumores mamários, combinando eficientemente as vantagens das CNNs para extração de características locais com a capacidade dos Transformers de capturar relações globais. A implementação das funções `TransformerEncoderLayer` e `TransformerEncoder` é fundamental para o sucesso da arquitetura, permitindo que o modelo aprenda representações ricas e contextualizadas das imagens médicas.

Esta arquitetura híbrida demonstra particular eficácia em aplicações médicas, onde tanto detalhes locais quanto padrões globais são cruciais para um diagnóstico preciso e confiável.
