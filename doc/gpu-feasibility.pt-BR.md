# Viabilidade de GPU no ART: CUDA, Tensor Cores e LightCraft

Análise inicial em 26/10/2026. As seções abaixo registram a investigação da revisão original, antes da implementação.

**Atualização:** a primeira etapa agora inclui instrumentação, um protótipo CUDA opcional para deconvolução e testes com RAWs locais. A implementação, as medições e os limites estão em [CUDA seletivo para deconvolução](cuda-deconvolution.pt-BR.md). A aceleração continua desativada por padrão.

**Segunda etapa:** foram testados [exibição OpenGL, cache em VRAM e Lanczos CUDA](gpu-display-cache-resize.pt-BR.md). O cache trouxe ganhos em cenários de reutilização; upload por operação e exibição da prévia pequena perderam para a CPU. Essas novas rotas permanecem experimentos isolados, sem ativação no editor ou na exportação.

**OpenCL testado:** a [comparação direta entre CPU, CUDA e OpenCL](opencl-feasibility.pt-BR.md) mostrou ganho OpenCL frente à CPU na deconvolução, mas desempenho semelhante ao CUDA na A4500. A troca de API não resolveu o custo das transferências. Para acelerar o uso real, a prioridade indicada pelas medições continua sendo o denoise e a reutilização de intermediários; o protótipo OpenCL permanece separado do aplicativo.

**Etapa avançada:** foram implementados e medidos [ajustes multicore, deconvolução CUDA recursiva, execução simultânea CPU/GPU e dois streams](advanced-performance.pt-BR.md). O relatório compara exportações reais, consumo de recursos, precisão e os experimentos que perderam para a CPU. Os novos caminhos continuam selecionáveis, com a implementação original disponível.

**Abertura e uso cotidiano:** implementados [decode RAW paralelo, prévia progressiva, cache limitado, escalonamento adaptativo e monitor CPU/GPU/RAM/VRAM](editor-performance.pt-BR.md), com medições de duplo clique, navegação, exportação e validação sem CUDA. O preset passou a detectar recursos de CPU automaticamente.

## Conclusão

É viável adicionar aceleração de GPU a este fork. Na revisão original não havia uma opção que ativasse processamento CUDA. O protótipo atual cobre a deconvolução descrita acima; ampliar a cobertura exige portar operações e integrar memória, cache, invalidação e tratamento de falhas ao pipeline existente.

**Diretriz acordada:** aceleração seletiva, somente onde houver ganho real. O pipeline pode continuar híbrido permanentemente: CPU → bloco GPU → CPU. Não há objetivo de portar tudo nem de manter a imagem na GPU de ponta a ponta. A unidade de decisão é uma operação ou um conjunto contíguo de operações, incluindo todos os custos de entrada e saída.

Para esta máquina NVIDIA, recomendo um backend **CUDA opcional em C++**, preservando o caminho CPU e os algoritmos do ART. Para uma iniciativa que priorize AMD/Intel/macOS, vale escolher um backend portátil antes dos ports: wgpu com uma ponte C/Rust permitiria aproveitar shaders WGSL, mas adicionaria outra linguagem e toolchain. Vulkan direto também exige uma implementação própria de infraestrutura e shaders.

O principal reaproveitamento do LightCraft é sua arquitetura e suas primitivas de computação. Seu renderizador completo não é um substituto compatível para o motor do ART. Tensor Cores são uma segunda frente: inferência de um modelo de redução de ruído, com contrato de entrada próprio, e não um interruptor que acelera todos os filtros atuais.

## Revisões e validação local

- ART: `4a7ed4fb775f555d54787a2748d923ed5c9a9223`, remoto `pjdgilmour/ART`.
- LightCraft: `bb08a697a41b2561a54f1b740245119503b24758`, cópia de leitura em `/tmp/art-lightcraft-review`.
- GPU consultada fora do sandbox: NVIDIA RTX A4500, compute capability **8.6**, driver **615.71.09**, memória reportada **19190 MiB**.
- Toolkit: `/usr/local/cuda/bin/nvcc`, CUDA **13.4**. `nvcc` não estava no PATH.
- O primeiro `nvidia-smi`, dentro do sandbox, falhou. Fora dele funcionou; isso não indicava defeito no driver.
- Compilei e executei um diagnóstico CUDA independente, `-arch=sm_86`, com 72 milhões de amostras FP32 (equivalente a RGB de 6000 × 4000). Fez upload, aplicou `max(x * 2 - 2000, 0)`, sincronizou, baixou e comparou todas as amostras com a fórmula na CPU. Resultado: **zero divergências** e nenhuma falha CUDA.

O diagnóstico é uma especialização simples da fórmula de exposição, em dados sintéticos finitos, incluindo negativos e valores acima de 65535. **Não valida o ART integrado, outras exposições, Tensor Cores nem desempenho de processamento RAW.** Fonte e executável temporários: `/tmp/art-cuda-probe.cu` e `/tmp/art-cuda-probe`. Reprodução:

```sh
/usr/local/cuda/bin/nvcc -O2 -arch=sm_86 /tmp/art-cuda-probe.cu -o /tmp/art-cuda-probe
/tmp/art-cuda-probe
```

Nenhum arquivo de implementação ou configuração de build do ART foi alterado. Não compilei o ART nem executei a suíte do LightCraft. Não há corpus RAW neste checkout; ganhos e fidelidade no produto ainda precisam de medição.

## Como o ART está organizado

| Evidência no checkout | Implicação para GPU |
|---|---|
| `CMakeLists.txt:145`, `:150`, `:501`: benchmark opcional e OpenMP; `rtengine/CMakeLists.txt` lista fontes CPU | Não há backend CUDA/OpenCL/Vulkan no motor examinado. Aproveitar instrumentação existente antes de escolher gargalos. |
| `rtengine/improcfun.cc:567`: `ImProcFunctions::process`, quatro estágios | Ponto de despacho compartilhado; manter ordem, parâmetros, condições e retornos de parada. |
| `rtengine/simpleprocess.cc`: exportação; `improccoordinator.cc`: navegador; `dcrop.cc`: preview detalhado | Acelerar somente exportação não acelera automaticamente todos os caminhos interativos. |
| `improccoordinator.cc:502` e `rtengine/refreshmap.h`: buffers e invalidação por evento | Já existe cache CPU. Estendê-lo com resultados residentes na GPU, sem recalcular estágios independentes do controle alterado. |
| `rtengine/imagefloat.h:35` e `rtengine/iimage.h:590` | FP32 planar, faixa nominal 0–65535, padding por linha; transportar strides e metadados. Não interpretar como RGB intercalado nem presumir faixa estrita. |
| `Imagefloat::Mode` | Os planos também representam XYZ, YUV e Lab. Um kernel deve declarar e respeitar modo, espaço de cor e escala. |
| `rtengine/ipexposure.cc:32` | Exposição já usa OpenMP/SSE2. Um kernel isolado é fácil, mas transferência pode custar mais que a operação. |

O pipeline organiza dehaze/compressão dinâmica no estágio 0; mixer/exposição/HSL/equalizador tonal no 1; nitidez e correções locais no 2; textura, curvas, simulação de filme e várias operações de cor no 3. Demosaic e denoise também têm caminhos anteriores a esses estágios. Portanto, substituir apenas `process()` não cobre todo o processamento.

## O que aproveitar do LightCraft

Links abaixo estão fixados na revisão analisada.

| Componente | Reaproveitamento proposto | Adaptação necessária |
|---|---|---|
| [render.rs / GpuStages](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/src/render.rs) | Fonte e intermediários mantidos na GPU; cache por estágio e vista; submissões limitadas | Integrar à invalidação, escala, recortes, máscaras e perfil de cor do ART. |
| [ctx.rs](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/src/ctx.rs) | Pool de buffers, limites de memória, sincronização e diagnóstico | Implementar equivalente CUDA; não reciclar memória ainda em uso. O código atual define pool padrão de 256 MiB, ajustável, apesar de números antigos no documento de GPU. |
| [blur.wgsl](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/src/wgsl/blur.wgsl) e [map.wgsl](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/src/wgsl/map.wgsl) | Passes de box blur, estatísticas e etapas de guided filter | Traduzir para CUDA se esse for o backend; preservar bordas, subsampling, raios, epsilon e matemática do ART. Gaussianas e variantes do guided filter não são automaticamente equivalentes. |
| [finish.wgsl](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/src/wgsl/finish.wgsl) | Fundir operações por pixel para reduzir tráfego de memória | Reescrever com fórmulas, LUTs e ordem do ART. Não importar sua ciência de cor como se fosse equivalente. |
| [lib.rs](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/src/lib.rs) e [testes](https://github.com/storytold/lightcraft/tree/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/tests/it) | CPU como referência, fallback explícito, falhas injetadas e diagnóstico de backend | Testar também FP32 e exportação 16 bits; restaurar entrada válida antes de repetir na CPU. |

O LightCraft usa **Rust + wgpu 30 + WGSL**, com backends Metal/Vulkan/DX12. Seu crate GPU depende dos tipos e parâmetros de vários outros crates do projeto; não oferece aqui uma biblioteca C++ pronta para ligar ao ART.

Uma ressalva decisiva está em `crates/gpu/src/lib.rs:453`: `handles()` aceita `OutputDepth::U8`, exclui soft proof com perfil de dispositivo e a visualização de faixa HDR. Os intermediários são float, mas a saída desse renderizador é de 8 bits. Exportações profundas usam CPU. Os testes de equivalência documentados comparam sRGB de 8 bits; isso não prova fidelidade suficiente para exportação de 16 bits/float do ART. A afirmação ampla de aceleração no README deve ser lida com esses limites.

## Prioridade dos ports

Estas prioridades são estimativas de engenharia; ainda não há perfil de desempenho do ART nesta máquina. A medição dos gargalos deve definir o primeiro port de produção. Mixer/exposição são opções de diagnóstico da infraestrutura, não uma etapa obrigatória antes de acelerar um filtro pesado.

| Trabalho | Potencial | Esforço relativo | Recomendação |
|---|---|---|---|
| Exposição + mixer, depois curvas/LUTs e conversões compatíveis | Resposta interativa; ganho depende de residência e fusão | Baixo por kernel; médio para integração | Diagnóstico opcional; integrar em produção somente com ganho medido, isoladamente ou junto de um bloco GPU já útil. |
| Guided filter, box blur e primitivas usadas por dehaze/equalizador tonal/textura | Vários passes e intermediários tornam a GPU interessante | Médio/alto | Primeiro alvo pesado após medir; compartilhar primitivas entre ferramentas. |
| Gaussianas, nitidez/deconvolução e resize | Paralelismo por pixel e por vizinhança | Médio/alto | Portar preservando filtros, bordas e máscaras do ART. |
| Denoise tradicional (`ipdenoise.cc`, `FTblockDN.cc`, wavelets) | Pode consumir parte relevante do tempo | Alto | Frente própria. Há wavelets e DCT FFTW; cuFFT não é substituto direto para a DCT real-to-real usada aqui. |
| Demosaic Bayer/X-Trans e reconstrução de altas luzes | Melhora abertura e exportação quando dominantes | Alto/muito alto | Posterior: múltiplos algoritmos e grande sensibilidade a artefatos e bordas. |
| Denoise neural com Tensor Cores | Grande carga de convoluções/matrizes | Alto; recurso novo | Projeto separado, com modelo e validação visual específicos. |

Em particular, contraste local do ART usa wavelets em `iplocalcontrast.cc`; portar o filtro de clareza do LightCraft no lugar dele mudaria o resultado. O objetivo deve ser acelerar a operação existente.

## Arquitetura CUDA recomendada

1. **Build opcional**: adicionar futuramente uma opção como `ART_ENABLE_CUDA`, desativada por padrão, e um target CUDA isolado. O projeto atual usa CMake mínimo 3.9 e flags globais C/C++11; selecionar e validar uma versão mínima adequada ao mecanismo CUDA adotado, escopando flags por linguagem. A compilação CPU deve continuar sem toolkit.
2. **Contexto e imagem GPU**: criar contexto persistente, buffers FP32 planares com stride explícito, pool limitado e estado de validade CPU/GPU. Inicialização fora da thread da interface; controle de concorrência entre preview e exportação.
3. **Bloco inicial**: escolher pelo perfil uma operação pesada ou um conjunto contíguo; fazer upload na entrada e download na saída, mantendo os intermediários desse bloco na GPU. Mixer/exposição podem servir de diagnóstico opcional. Ampliar o bloco somente quando isso melhorar o tempo observado e justificar a complexidade adicional.
4. **Cache**: vincular intermediários GPU às mesmas causas de invalidação do ART. Mudanças de imagem, crop, escala, parâmetros e perfis precisam invalidar as entradas corretas. Leitores CPU, histogramas e pipetas precisam de sincronização explícita.
5. **Falhas**: erros de alocação/execução devem retornar a uma entrada CPU íntegra e refazer o trabalho; nunca entregar um estágio parcialmente modificado. Registrar motivo do fallback e permitir desativação manual. Erro assíncrono deve ser verificado antes de publicar o resultado.
6. **Precisão**: manter FP32 nas operações fotográficas e as LUTs/fórmulas do ART. Não ativar fast-math globalmente nem converter todo o pipeline para FP16. Mesmo a faixa nominal 65535 ultrapassa o máximo finito FP16 de 65504; normalizar resolve esse caso de faixa, mas não a perda de precisão.

Uma imagem RGB FP32 de 24 MP ocupa 288 MB decimais, aproximadamente 275 MiB, sem padding. Em 60 MP são 720 MB, aproximadamente 687 MiB, por imagem. Entrada, saída, caches, máscaras e scratch multiplicam isso. A A4500 permite experimentar com folga relativa, mas o orçamento deve usar memória disponível e considerar outros processos.

O ganho depende de evitar sucessivas idas CPU↔GPU. A NVIDIA recomenda minimizar essas transferências e manter dados intermediários no dispositivo em seu [guia de boas práticas CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html). É por isso que exposição sozinha não é uma boa demonstração do ganho final.

## Tensor Cores e denoise por IA

Tensor Cores executam operações matriciais; convoluções de redes podem aproveitá-los via bibliotecas apropriadas. Curvas, LUTs, medianas, guided filters e demosaic tradicionais não passam a usá-los por simplesmente compilar com CUDA. Transformações RGB 3 × 3 podem ser expressas matricialmente, mas empacotamento, tamanho reduzido e tráfego de memória tornam isso um alvo inicial pouco convincente.

O [kernel neural do LightCraft](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/crates/gpu/src/wgsl/nn_conv.wgsl) implementa convolução em blocos com acumuladores `vec4<f32>` e memória de workgroup. Não encontrei nesse caminho invocações explícitas de operações Tensor Core/cooperative matrix. Não se deve tomar o uso de GPU desse executor como evidência de uso dos Tensor Cores.

Para explorá-los no ART, uma opção é um modelo ONNX executado por TensorRT, com CUDA/CPU para operações não atendidas; a [documentação do ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) recomenda registrar também o provedor CUDA. Seleção de precisão, suporte de operadores e uso efetivo dos Tensor Cores precisam ser medidos; não são garantidos apenas pela escolha do provedor. Compatibilidade entre runtime, driver, toolkit e bibliotecas deve ser validada como conjunto.

A [implementação de denoise do LightCraft](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/docs/denoise.md) oferece ideias úteis de tiles com sobreposição, cache por fonte/modelo e comparação CPU/GPU. Seu contrato atual recebe Bayer RGGB normalizado, com preto subtraído e sem white balance, e produz RGB de câmera. É uma operação anterior ao restante do desenvolvimento, incluindo demosaic neural; não pode ser inserida diretamente no denoise RGB existente. Não cobre X-Trans nesse contrato. Os pesos são separados do programa e têm licença própria.

A [referência NVIDIA de precisão mista](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) explica operações matriciais e limites numéricos. Para fotografia, a aceitação deve incluir sombras, textura fina, altas luzes e cor, além do erro médio de tensores.

## Reaproveitamento de código e licenças

O ART examinado traz GPL-3.0-ou-posterior nos arquivos do motor. O LightCraft oferece o núcleo sob MIT ou Apache-2.0, conforme [LICENSE-MIT](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/LICENSE-MIT) e [NOTICE](https://github.com/storytold/lightcraft/blob/bb08a697a41b2561a54f1b740245119503b24758/NOTICE). Isso permite considerar ports dos componentes cobertos por essas licenças para o ART, preservando atribuições e condições aplicáveis; a [FSF lista MIT/Expat como compatível com GPL](https://www.gnu.org/licenses/license-list.html), e a [Apache confirma compatibilidade de Apache-2.0 com GPLv3](https://www.apache.org/licenses/GPL-compatibility.html).

Verificar o cabeçalho e a origem de cada arquivo efetivamente portado. A licença do núcleo não se estende automaticamente a modelos, marcas e todos os componentes de terceiros. Nenhum código do LightCraft foi incorporado nesta análise.

## Como decidir se vale integrar

- Para cada bloco candidato, comparar `T_CPU` com `T_preparo + T_upload + T_GPU + T_download + T_sincronização`, incluindo alocação quando necessária. Medir o trecho completo por relógio de parede para não omitir custos nem contar duas vezes esperas sobrepostas. Tempos de kernel servem para diagnóstico, não para decidir a integração sozinhos.
- Exigir ganho repetível acima da variabilidade das medições e redução relevante no tempo da tarefa percebida pelo usuário, preservando qualidade. Registrar repetições, mediana e dispersão; distinguir primeira execução e execuções com cache. Um filtro muito acelerado pode ter impacto pequeno se consumir pouco tempo no fluxo completo.
- Definir a seleção CPU/GPU por operação e condições medidas, como resolução, raio, qualidade e disponibilidade de memória. Um filtro pode compensar na exportação e continuar na CPU em previews pequenos. Os limiares serão derivados dos benchmarks; não presumir que GPU é sempre mais rápida quando disponível.
- Medir CPU com `WITH_BENCHMARK=ON` e complementar a instrumentação por estágio; separar abertura RAW, preview frio, arraste de controles, zoom 100% e exportação. Usar Release com OpenMP, não uma CPU artificialmente limitada.
- Comparar CPU/GPU no mesmo arquivo e perfil, em imagens Bayer e X-Trans, 24–60 MP ou resoluções representativas do uso real. Medir também tempo total com transferências, inicialização, cache frio/quente e pico de VRAM.
- Comparar imagens intermediárias FP32 e saídas 16 bits/float; erro absoluto/relativo, finitude, cor e inspeção visual. Definir tolerâncias por operação, especialmente para somatórios, reduções e funções transcendentais. O limiar de 8 bits do LightCraft não basta.
- Cobrir bordas, padding, recortes, escalas, máscaras, highlights fora da faixa nominal, perfis ICC/DCP e pipetas/histogramas. Verificar que cache quente coincide com recomputação limpa.
- Testar build sem CUDA, GPU ausente, erro de inicialização, VRAM insuficiente, cancelamento e falha durante execução, preservando a saída CPU correta.

**Próximo passo recomendado:** medir os gargalos e selecionar um bloco com potencial de ganho líquido; implementar apenas a infraestrutura CUDA opcional necessária a esse bloco. Guided filter/box blur são candidatos, sujeitos ao perfil. Manter na CPU as operações e condições sem benefício comprovado. Tensor Cores permanecem uma possibilidade posterior para um recurso neural separado. O teste local prova que a máquina executa CUDA; só o protótipo integrado e o corpus fotográfico poderão quantificar o benefício no ART.
