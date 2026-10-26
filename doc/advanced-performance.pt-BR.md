# CPU multicore, CUDA e execução híbrida no ART

Investigação prática em 26/10/2026, sobre `4a7ed4fb775f555d54787a2748d923ed5c9a9223`. Continua os testes de [deconvolução](cuda-deconvolution.pt-BR.md), [exibição/cache/resize](gpu-display-cache-resize.pt-BR.md) e [OpenCL](opencl-feasibility.pt-BR.md).

**Continuação:** [abertura progressiva, decode RAW paralelo, cache, política adaptativa e monitor de hardware](editor-performance.pt-BR.md). Essa etapa atualiza o preset; os números abaixo preservam a configuração histórica usada nesta rodada.

## Ambiente e método

- Dois Xeon E5-2680 v4: 28 núcleos físicos, 56 threads; RTX A4500, compute capability 8.6, 19.190 MiB de VRAM informados pelo driver.
- Debian 13, GCC 14.2, CUDA 13.4, driver 615.71.09; Release, OpenMP, `-march=native`, sem fast-math e sem contração FMA nas comparações.
- RAWs e sidecars fornecidos em `/home/paulo/Imagens/2026/20261011/`, apenas lidos. Três casos: `276A0950` sem denoise, `276A1012` com denoise normal e `276A1020` com denoise agressivo, NLMeans e reforço de nitidez nos cantos. Os três processam 6732 × 4492 pixels; os dois últimos exportam 2560 × 1708.
- Os perfis têm versão 1045 e este fork usa 1043. Todas as variantes usam o mesmo leitor, arquivos e perfis; não se presume equivalência com a versão mais nova do ART.
- Exportações sequenciais, configurações/cache do ART separados por execução, variantes e fotos em ordem rotativa. O cache de arquivos do sistema operacional não foi esvaziado. Não houve compilação simultânea às medições.
- `tools/benchmark_art.py` registra comandos, ambiente da variante, duração do processo, eventos internos e amostras de recursos. A rodada final usa três repetições. Triagens anteriores, identificadas separadamente, têm uma ou duas repetições.
- RAM é o maior `VmHWM` observado em `/proc`. CPU é tempo acumulado amostrado dividido pelo tempo de parede; pode subestimar o fim do processo. VRAM do processo é filtrada pelo PID; utilização GPU é do dispositivo inteiro, amostrada por `nvidia-smi`, não ocupação de SMs. Os números incluem o custo da amostragem.
- Testes de kernels usam também 1,05, 4,37, 8,39 e 16,38 MP, bordas e strides irregulares. Não há RAW X-Trans nem um conjunto de sensores/resoluções nativas diferentes nesta validação.

## O que foi implementado

### CPU

`ART_CPU_OPT=1` paraleliza a máscara de recuperação de detalhes, que era calculada com `multithread=false`; limita a equipe de análise de ruído às nove amostras existentes; e dimensiona o pool auxiliar a partir do limite OpenMP configurado.

`ART_CPU_THREADS=1` faz o denoise respeitar `OMP_NUM_THREADS` e a opção `multiThread`, em vez de criar equipes pelo total de processadores lógicos. Isso também reduz buffers temporários por thread. A varredura 8/14/28/56 mostrou que usar 56 threads não foi a melhor escolha nesta máquina.

`ART_EARLY_IMAGE=1` antecipa a geração da imagem completa, antes das nove amostras de ruído. O primeiro `getImage` pode executar recuperação de altas luzes. Na ordem antiga, isso acontece dentro da região paralela das amostras e pode serializar trabalho interno. A nova ordem conserva as chamadas e evita esse contexto aninhado; aplica-se à exportação, não reordena o coordenador de preview.

`ART_DENOISE_ORDERED=1` distribui linhas de blocos DCT por grupos sem sobreposição, com barreiras entre grupos. No código original, linhas adjacentes podem acumular simultaneamente em `Ldetail` e `totwt`. A nova distribuição elimina essa disputa específica. A ordem das somas muda; ela não torna todo o pipeline determinístico.

`ART_FFTW_ESTIMATE=1` foi implementado e ensaiado como alternativa ao planejamento FFTW_MEASURE. O planejamento medido custou apenas cerca de 23–25 ms por exportação; a variante não é necessária na configuração principal.

### CUDA recursivo

A deconvolução passa a aceitar, com `ART_CUDA_RECURSIVE=1`, sigma acima de 1,15 até 4,0. O port reproduz a recorrência Young–van Vliet, condições de contorno e diferenças de precisão dos caminhos SSE e suas caudas. Duas transposições com memória compartilhada permitem acesso coalescido; as vinte iterações mantêm os intermediários na VRAM. Não substitui o filtro recursivo por uma convolução aproximada.

O backend continua chamável diretamente em imagens menores para testes, mas o motor só despacha o caminho recursivo a partir de 8.388.608 pixels. O limite deriva desta máquina: 4,37 MP perdeu para a CPU; 8,39 MP ficou próximo do empate; 16,38 e 30,24 MP ganharam. Isso não é uma calibração universal para outras GPUs/CPUs.

O pool recursivo usa sete planos FP32 e um de bytes: aproximadamente 836 MiB em 30,24 MP, além do contexto e bibliotecas. O limite de 64 MiPixels por requisição continua valendo. O armazenamento host usa dois planos FP32 e um de bytes. As alocações são reaproveitadas; o conteúdo de entrada é atualizado a cada chamada.

`ART_CUDA_PINNED=1` permite staging host page-locked, com fallback para memória comum se a reserva falhar. É um experimento separado: reduz transferências com buffers reutilizados, mas a primeira reserva tem custo e fixa RAM física. Não é ligado automaticamente.

### CPU e GPU simultâneas

O reforço de nitidez nos cantos calcula duas estimativas independentes, para sigma e sigma + delta. `ART_HYBRID=1` executa uma na CPU/OpenMP e outra em uma thread que alimenta o CUDA. A versão refinada reserva quatro threads para preparar a entrada GPU e 24 para o filtro CPU quando o limite total é 28. O limite original da thread principal é restaurado ao sair do trecho; `ART_HYBRID_WORKERS=1` permite reproduzir o primeiro experimento, com cópia serial. A combinação dos resultados só ocorre depois da conclusão de ambas.

`ART_GPU_PAIR=1` testa as duas estimativas na GPU, em streams e pools separados. O máximo é dois pools, alocados sob demanda; se ambos estiverem ocupados, a chamada usa CPU. Essa alternativa pode sobrepor kernels e transferências, mas dobra o orçamento de buffers para o par. Quando ambas as flags estão presentes, `ART_GPU_PAIR` prevalece.

As variantes são opcionais, usam o motor compartilhado por CLI e GUI e preservam o caminho original. Falhas CUDA não publicam uma imagem parcialmente atualizada. Falha ao criar a thread volta ao despacho sequencial. GPU indisponível e recusa de uma chamada também são cobertas por testes.

### Wavelets: protótipo testado e rejeitado para despacho atual

`tools/gpu-pipeline-tests` implementa cinco níveis Haar, mantendo a pirâmide na VRAM e comparando todos os coeficientes com `cplx_wavelet_level.h`. O teste 3366 × 2246 corresponde à dimensão após o primeiro nível da imagem de 30 MP; não é um port completo do denoise.

| Execução | CPU | Upload + kernels + download |
|---|---:|---:|
| RAM comum, buffers GPU reutilizados | cerca de 44 ms | cerca de 71 ms |
| RAM registrada/pinned | cerca de 43 ms | cerca de 51 ms |

Os kernels isolados levam aproximadamente 1,7 ms. A transferência dos quinze planos de detalhes e do lowpass domina; registrar a RAM custou outros 13 ms uma vez. Todos os coeficientes coincidiram exatamente nos testes. O caminho não foi integrado ao motor porque perde quando os consumidores seguintes continuam na CPU. Uma continuação útil exige portar também as operações que consomem esses coeficientes.

## Medições e qualidade

Foram feitas **69 exportações nas rodadas finais de desempenho**, além das triagens e verificações de precisão. Resultados completos: [JSON dos benchmarks](benchmarks/advanced-performance-20261026.json). Tempos abaixo são medianas de três execuções, em segundos; menores são melhores.

| Foto / processamento normal em 30,24 MP | CPU original | CPU otimizada | CPU otimizada + CUDA sequencial | Híbrido inicial | Dois streams |
|---|---:|---:|---:|---:|---:|
| 276A0950, sem denoise | 4,966 | 5,007 | 4,905 | 4,843 | 4,869 |
| 276A1012, denoise normal | 13,206 | 10,382 | **9,913** | 10,017 | 10,047 |
| 276A1020, denoise agressivo + cantos | 13,272 | 11,444 | 11,451 | **11,155** | 11,407 |

Na 1012, CPU otimizada + CUDA reduz o tempo em **24,9% (1,33×)**. O denoise passa de 4,99 para aproximadamente 2,39 s; a nitidez, de 0,59 para 0,27 s. Na 0950, a nitidez cai de 0,55 para 0,28 s, mas o processo completo ganha somente cerca de 1–2,5%, dentro de uma dispersão relevante. O carregamento inicial de aproximadamente 2,6–2,7 s e os demais custos limitam o ganho global.

Na 1020, o híbrido inicial reduz a exportação em **16,0% (1,19×)**. Dois streams não trouxeram vantagem sobre o híbrido e consumiram mais VRAM. As variantes de par não são acionadas nas duas primeiras fotos; diferenças pequenas entre suas colunas são variação entre execuções.

Uma segunda rodada, também com três repetições e ordem rotativa, refinou a distribuição de CPU no par da 1020:

| Variante | Mediana total | Mín.–máx. | Par de deconvoluções, mediana |
|---|---:|---:|---:|
| CPU otimizada, controle da rodada | 11,623 | 11,465–11,735 | duas chamadas sequenciais |
| Híbrido, staging em uma thread | 11,061 | 10,928–11,066 | 0,893 |
| **Híbrido, quatro threads de staging + 24 CPU** | **10,655** | **10,577–10,991** | **0,782** |
| Híbrido 24+4 com staging pinned | 10,860 | 10,753–11,117 | 0,827 |

O híbrido refinado ganha **8,3% sobre a CPU já otimizada na mesma rodada**. Comparado aos 13,272 s da rodada anterior, corresponde a uma redução indicativa de **19,7% (1,25×)**; essa última comparação é entre rodadas. O packing CUDA cai de cerca de 240 para 84 ms. Pinned acelera transferências isoladas, mas não ganhou na exportação de uma foto em processo novo, por isso não integra a configuração indicada.

### Resolução e exportação rápida

O caminho recursivo foi medido separadamente, com contexto já inicializado, sigma 1,27 e comparação FP32 exata:

| Dimensões | CPU | CUDA total |
|---|---:|---:|
| 1025 × 1027 | aproximadamente 25 ms | aproximadamente 53 ms |
| 2560 × 1708 | 85,9 ms | 133,3 ms |
| 4096 × 2048 | 215,1 ms | 204,4 ms |
| 5120 × 3200 | 431,6 ms | 313,4 ms |
| 6732 × 4492 | 803,1 ms | 610,6 ms |

A primeira chamada fria dos testes isolados inclui centenas de milissegundos de inicialização; no ART a preparação do contexto ocorre em segundo plano durante a abertura. Os tempos de exportação incluem essa abertura.

Também foram feitas exportações reais com `-f`, comparando original e otimizado **dentro do mesmo pipeline rápido**:

| Dimensão de processamento observada | Original | Otimizado | Redução |
|---|---:|---:|---:|
| 1920 × 1281 (2,46 MP) | 6,450 s | 5,212 s | 19,2% |
| 1280 × 854 (1,09 MP) | 6,078 s | 5,019 s | 17,4% |

O primeiro perfil pedia 2560 pixels, mas as opções padrão de exportação rápida limitaram a imagem a 1920; a tabela registra as dimensões observadas no motor. Em 1,09 MP a nitidez permaneceu na CPU. Não interpretar a comparação normal × rápida como ganho de implementação: a ordem e a escala do processamento são diferentes.

### Uso de recursos

Amostras da foto 1020; RAM é pico observado e VRAM inclui o contexto CUDA:

| Variante | RAM | VRAM do ART | CPU média, equivalentes a threads ocupadas |
|---|---:|---:|---:|
| CPU original | 6513 MiB | 0 | 15,2 |
| CPU otimizada | 5969 MiB | 0 | 13,4 |
| CUDA sequencial | 6176 MiB | 1022 MiB | 10,5 |
| Híbrido inicial | 6115 MiB | 1022 MiB | 12,6 |
| Dois streams | 6235 MiB | 1862 MiB | 10,5 |
| Híbrido refinado, rodada adicional | 6117 MiB | 1022 MiB | 12,7 |

As leituras de utilização GPU chegaram a 100% nos caminhos recursivos. Contudo, houve atividade do dispositivo também durante execuções CPU; esse contador inclui outros clientes gráficos e **não deve ser atribuído exclusivamente ao ART**. O JSON preserva essas amostras. VRAM filtrada por PID e os tempos dos blocos são medidas mais úteis aqui. Menos threads/menor utilização média com menos tempo de execução é um ganho, não uma regressão.

### Fidelidade e limitações observadas

- Oito testes CUDA passaram: deconvolução direta, recursiva, resize, par híbrido, dois streams e caminhos sem dispositivo. Os testes numéricos incluem bordas, padding, dados fora da faixa nominal e entradas não finitas/fallback.
- O teste separado Haar e os quatro testes OpenCL também passaram depois da recompilação da referência CPU. CLI compilado com e sem CUDA; interface gráfica compilada com CUDA.
- Nos kernels recursivos testados em cinco resoluções, a diferença FP32 foi zero. Na verificação sobre entradas reais, sigmas **0,348989695, 1,27 e 1,31** também produziram **zero amostras diferentes e zero valores não finitos**. O par híbrido sintético coincidiu exatamente com as duas chamadas CPU.
- O pipeline completo CPU já não é determinístico. Na rodada final, repetir a CPU original produziu erro médio de aproximadamente 0,0000123, 0,00538 e 0,822 LSB de 16 bits nas fotos 0950, 1012 e 1020, respectivamente. Comparações com as variantes ficaram na mesma ordem: aproximadamente 0,000013, 0,0052–0,0056 e 0,821–0,831 LSB.
- Na exportação rápida, a CPU original também variou entre repetições: erro médio de 7,127 LSB em 1920 × 1281 e 3,150 LSB em 1280 × 854. Original × otimizado ficou em 7,113 e 3,166 LSB, respectivamente. Esses resultados não comprovam determinismo do pipeline rápido.
- Máximos pontuais são maiores: na 1020 ocorreram diferenças de milhares de LSB, inclusive entre execuções CPU. A distribuição DCT sem disputa não eliminou toda a variação. Uma rodada preliminar teve um desvio sistemático muito maior (cerca de 56,6 LSB médios), que não se repetiu nas rodadas de isolamento e finais; sua causa não foi estabelecida. **Não se afirma equivalência bit a bit do denoise completo nem validação fotográfica universal.**
- A inspeção de recortes da foto 1020 e da abertura no editor não mostrou uma mudança estrutural aparente; é uma verificação limitada, não substitui o exame de todo o corpus. Os recursos novos permanecem opt-in por esse motivo e pela calibração restrita ao hardware/casos medidos.

### Interface

O teste `tools/probe_art_gui.py` abriu cópias do RAW e sidecar em Xvfb, com configurações isoladas, e capturou a janela. Foi necessário corrigir a disposição dos ícones nos dados temporários; depois disso, ambas as variantes abriram e concluíram o preview. O harness encerra o processo depois de 25 s; não é um teste de encerramento normal.

| Medida, uma abertura por variante | Original | Otimizado |
|---|---:|---:|
| Atualização inicial de preview | 559 ms | 551 ms |
| Primeiro crop do editor | 1627 ms | 1589 ms |
| Intervalo mediano do callback de 16 ms | 16,119 ms | 16,099 ms |
| Percentil 95 do intervalo | 16,135 ms | 16,129 ms |
| Primeiro callback, incluindo inicialização da janela | 4903 ms | 4770 ms |

O preview era pequeno e ficou na CPU. Não há evidência de ganho relevante de responsividade nesse ensaio, nem medição de arraste contínuo, zoom 100% ou apresentação no compositor real. A maior pausa ocorreu na inicialização da janela. A aceleração de exportação não deve ser apresentada como uma melhora equivalente da interface.

## Operações inspecionadas e decisões

| Área | Evidência e decisão nesta etapa |
|---|---|
| Leitura RAW e demosaic | Instrumentação de carregamento, preprocessamento e demosaic; demosaic ficou em cerca de 0,09 s no caso leve. Não foi portado. Separar carregamento/descompressão do demosaic é essencial para não atribuir todo o tempo de abertura ao algoritmo errado. |
| Denoise/wavelets/DCT | Máscara multicore, limites de equipes, antecipação da imagem, distribuição de blocos DCT e protótipo Haar CUDA implementados. NLMeans foi inspecionado; sua recorrência de integral por tiles e LUT precisam ser preservadas em um port futuro. |
| Nitidez | Gaussiana direta anterior e nova recursiva CUDA; par CPU/GPU e dois streams integrados como variantes selecionáveis. PSF arbitrária continua na CPU. |
| Cor, curvas, máscaras e ajustes locais | Instrumentados pelo pipeline; há operações individuais de poucos milissegundos e curvas de centenas de milissegundos. Não foi criado um caminho CUDA aproximado para substituir SLEEF/LUTs, ICC ou algoritmos locais. As máscaras existentes aproveitaram paralelismo CPU. |
| Resize e cache VRAM | Protótipos anteriores preservados. Resize residente pode ganhar; upload de uma imagem nova pode apagar o ganho. Não há cache de foto RGB inteira integrado ao editor. |
| Exibição | Protótipo GL anterior preservado; ganho em crop de textura grande, perda no preview pequeno. A GUI completa foi compilada para a validação desta etapa; teste de abertura não equivale a uma medição de arraste contínuo de controles. |
| OpenCL | Comparação anterior não justificou trocar CUDA na A4500. Backend permanece isolado nos testes. |
| Tensor Cores | Nenhum kernel desenvolvido usa Tensor Cores. Os algoritmos preservados usam FP32 e recorrências/filtros; inserir um denoise neural seria um recurso diferente e exigiria validação própria. |

## Configuração com melhor resultado global nesta máquina

A configuração medida nesta etapa usou 28 threads e par híbrido 24+4, além das otimizações CPU/CUDA descritas acima. O script [`tools/run_art_optimized.sh`](../tools/run_art_optimized.sh) foi posteriormente atualizado para detectar recursos e habilitar a abertura progressiva; consulte o [relatório de abertura e navegação](editor-performance.pt-BR.md). Para reproduzir o limite histórico, use `OMP_NUM_THREADS=28 ART_HYBRID_WORKERS=4` e a matriz desta etapa. Dois streams, pinned e FFTW_ESTIMATE continuam desligados por padrão; variáveis explícitas prevalecem.

Com um build instalado normalmente:

```sh
bash tools/run_art_optimized.sh /caminho/ART -N
# Ou exportação:
bash tools/run_art_optimized.sh /caminho/ART-cli -a -o /tmp/resultado.tif \
  -t -b16 -p /caminho/foto.cr2.arp -c /caminho/foto.cr2
```

`-N` abre uma instância gráfica nova, evitando enviar o pedido a uma instância que já esteja executando com outro ambiente. Para usar os binários temporários testados nesta sessão:

```sh
LD_LIBRARY_PATH=/tmp/art-build-deps/usr/lib/x86_64-linux-gnu \
  bash tools/run_art_optimized.sh /tmp/art-build-cpu/rtgui/ART -N
```

O wrapper não instala software nem altera as variáveis do shell que o chamou. Executar o binário sem o wrapper e sem essas variáveis mantém os caminhos de processamento originais. O monitor de hardware acrescentado na etapa seguinte aparece por padrão e pode ser ocultado com `ART_HW_MONITOR=0`. `ART_CUDA=0` no wrapper permite usar apenas as otimizações CPU. O preset permanece experimental; não altera perfis nem grava preferências para ativar as otimizações. A seleção de CPU agora é adaptativa, mas os ganhos publicados continuam restritos aos casos medidos.

## Reprodução

Build do motor e da interface, com as dependências normais do ART disponíveis:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DART_ENABLE_CUDA=ON -DWITH_BENCHMARK=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target art art-cli -j8
```

Testes CUDA e do protótipo de wavelets:

```sh
cmake -S tools/cuda-tests -B build-cuda-tests -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-tests -j8
ctest --test-dir build-cuda-tests --output-on-failure
cmake -S tools/gpu-pipeline-tests -B build-gpu-pipeline-tests -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-gpu-pipeline-tests -j8
OMP_NUM_THREADS=28 build-gpu-pipeline-tests/haar_test 3366 2246
OMP_NUM_THREADS=28 build-gpu-pipeline-tests/haar_test 3366 2246 pinned
```

`tools/benchmark_art.py --matrix matriz.json --cli /caminho/ART-cli --output /tmp/rodada-nova --repeat 3 --gpu-samples /caminho/foto.cr2` reproduz a comparação. A matriz usada está no JSON de resultados. `replay_matrices.final` explicita `ART_HYBRID_WORKERS=1` para reproduzir o comportamento anterior ao refinamento; a rodada `quota` testa o preset final com quatro workers. `--fast` seleciona o pipeline de exportação rápida; `--profile` permite um sidecar específico. Não comparar exportação normal com rápida como se fossem o mesmo algoritmo.

`ART_CUDA_VERIFY=1` compara CPU e GPU sobre **a mesma entrada de nitidez**, imprime erro máximo/médio e devolve a saída CPU. Nesse modo o par simultâneo fica desativado para permitir a comparação individual. Não usar esses tempos como benchmark. `ART_PROFILE=1` mede estágios inclusivos; não somar pai e filhos. `ART_UI_PROFILE=1` instala um callback de 16 ms no loop GTK e registra intervalos reais em stderr; callbacks não existem na execução normal.

Para repetir a abertura isolada (requer Xvfb e ImageMagick):

```sh
python3 tools/probe_art_gui.py --cli /caminho/ART --matrix matriz.json \
  --output /tmp/gui-nova --seconds 25 /caminho/foto.cr2
```

As dependências ausentes foram extraídas em `/tmp/art-build-deps`, sem instalar pacotes no sistema. Os executáveis desta sessão estão em `/tmp/art-build-cpu/rtgui/ART` e `ART-cli`; o nome da pasta é histórico, ambos foram compilados com CUDA. Dependem dos dados em `/tmp/art-runtime-data` e das bibliotecas extraídas, e não constituem uma instalação persistente.

A revisão final de esperas, memória, RAW multiframe, monitor e encerramento CUDA está em [Revisão adicional de estabilidade e desempenho](editor-performance.pt-BR.md#revisão-adicional-de-estabilidade-e-desempenho--26102026), com testes e limites documentados.

Para instalar a versão otimizada por APT, veja [Pacote Debian](optimized-deb.pt-BR.md). O pacote inclui lançadores com o preset ativado e dispensa os caminhos temporários desta sessão.
