# Abertura, navegação e escalonamento adaptativo no ART

Etapa de 26/10/2026. Dados, matrizes, eventos e amostras: [JSON completo](benchmarks/editor-performance-20261026.json). Continuação de [CPU/CUDA e processamento híbrido](advanced-performance.pt-BR.md). O controle desta etapa é o **preset otimizado anterior**, e não uma segunda comparação contra o ART original. As melhorias de CUDA, resize e deconvolução anteriores continuam disponíveis.

Ambiente: Debian 13, dois Xeon E5-2680 v4 (28 núcleos físicos/56 lógicos), RTX A4500, GCC 14.2 e CUDA 13.4. Mesmo conjunto de CR2 Dual Pixel da Canon 5D Mark IV e sidecars do relatório anterior; não foi obtida uma amostra de outros sensores. Configurações e cache do ART isolados por execução, sem esvaziar o cache de arquivos do sistema operacional e sem compilar durante as medições.

## Resultado principal

O gargalo novo confirmado foi a **serialização da decodificação dos dois frames dos CR2 Dual Pixel**. Cada frame já tinha um objeto LibRaw independente, mas `unpack()` e `raw2image()` passavam por um mutex global. A leitura do arquivo levou aproximadamente 28 ms por chamada e os metadados cerca de 1 ms; paralelizá-los não resolveria os aproximadamente 2,6 s do carregamento RAW.

Com `ART_RAW_PARALLEL=1`, a região existente de frames pode efetivamente decodificar em paralelo. O carregamento inicial caiu de **2,616 para 1,422 s**. Na exportação completa da foto 0950, a mediana caiu de **4,774 para 3,608 s**, redução de **24,4%**, em três repetições alternadas. A conversão RAW para planos levou cerca de 12 ms; a implementação AVX2 não melhorou esse gargalo de forma consistente.

O fork procura `libraw_r>=0.21`, a biblioteca com suporte a threads. A remoção opcional do mutex se apoia no uso de instâncias independentes, criadas durante `do_loadRaw`, e nas [condições de segurança da API LibRaw](https://www.libraw.org/docs/API-notes.html#threads). Isso não autoriza usar um mesmo objeto concorrentemente nem valida todos os decodificadores disponíveis.

## Teste real de duplo clique

`tools/probe_art_navigation.py` inicia ART com um diretório e envia duplos cliques via XTest em um **Xvfb privado**, cinco segundos após o lançamento. As três miniaturas já estão visíveis; atualizações de miniaturas com os perfis podem continuar em segundo plano. A sequência é 0950 → navegador → 1012 → navegador → 0950. Foram inspecionadas capturas do navegador, da prévia temporária e do editor final. Fotos e sidecars são copiados para `/tmp`; os originais não são gravados.

**Correção metodológica:** o ART salva no sidecar ajustes automáticos calculados durante a edição. Os dois pilotos que compartilhavam sidecars entre sessões foram descartados da comparação final. O harness agora restaura os perfis originais **antes de cada sessão** e registra seus SHA-256. Todos os hashes das 14 sessões finais coincidiram com os originais. Alterações durante a navegação dentro de uma sessão fazem parte do uso normal; não passam para a próxima variante.

Medições em segundos a partir de `FilePanel::fileSelected`, três sessões para controle/preset automático e duas para as alternativas. A janela tem aproximadamente 1200 × 680 pixels. “Processada” é o primeiro desenho real do crop do editor, com o perfil aplicado, na escala atual. Não significa exportação em resolução total nem ausência de atualizações posteriores.

| Ação | Controle anterior | Automático, dinâmico | Grupos fixos | Prévia da câmera, automático |
|---|---:|---:|---:|---:|
| Primeira abertura, 0950 | 4,023 | **2,049** | 2,287 | 0,255 |
| Alternância para 1012, denoise | 6,434 | **5,369** | 5,279 | 0,248 |
| Retorno à 0950 | 3,293 | **2,031** | 2,119 | **0,071**, cache |

As reduções medianas foram **49,1%, 16,6% e 38,3%**, respectivamente. O controle da primeira abertura variou de **3,335 a 4,421 s**; o automático, de **2,015 a 2,184 s**. Na foto com denoise, os intervalos foram 6,237–6,579 e 5,360–5,377 s. São amostras pequenas, com concorrência real de tarefas auxiliares e cache do sistema operacional aquecido, não garantias para qualquer foto/disco.

Separação dos marcos, primeira abertura:

| Marco | Controle | Automático |
|---|---:|---:|
| RAW entregue à interface | 3,428 s | 1,408 s |
| Primeiro `controls_ready` | 3,998 s | 2,017 s |
| Primeiro crop desenhado | 4,023 s | 2,049 s |
| Último sinal de conclusão/desenho antes de voltar ao navegador | 4,207 s | 2,226 s |

Na foto 1012, o último sinal de conclusão/desenho ocorreu em 6,517 / 5,437 s. `controls_ready` registra o estado de processamento da interface e pode ocorrer mais de uma vez, inclusive antes do desenho. O último sinal é uma aproximação da estabilização visual nesta sequência controlada. Nenhum deles mede diretamente a latência de um arraste de slider.

O callback do loop GTK solicita intervalos de 16 ms. Na primeira abertura, o p95 foi **16,137 / 16,161 / 17,818 ms**, controle/automático/grupos. Os maiores intervalos foram **156 / 169 / 145 ms**: não há melhora demonstrada na pior pausa com o automático. Durante a foto com denoise, seu p95 chegou a **18,744 ms**, contra 16,138 ms no controle. A melhora comprovada é concluir a abertura mais cedo; a fluidez não melhorou uniformemente. Xvfb não mede o compositor físico nem a latência do mouse na sessão do usuário.

As três miniaturas terminaram aproximadamente 2,5–2,8 s após o lançamento, incluindo inicialização; não houve vantagem consistente nessa pequena pasta. O pico de RSS durante a navegação ficou em **3,62–3,65 GiB** no controle, **3,71–3,74 GiB** no automático e **3,64–3,65 GiB** nos grupos. Não houve ensaio de um catálogo com milhares de arquivos.

### O que as alternativas realmente acrescentaram

Na rodada de duas repetições, manter a fila antiga e ligar somente as novas funções de abertura/cache/decode produziu 2,096 / 5,160 / 2,041 s para as três ações. Portanto, **a maior parte do ganho vem do decode paralelo e da prévia, não de uma superioridade geral comprovada do novo escalonador**. O preset usa a política dinâmica para adaptar os limites ao hardware e moderar trabalho de fundo, sem fixar 28 threads.

Prefetch reduziu a prévia da foto seguinte de aproximadamente 0,24 para **0,07 s**, com confirmação de cache hit. O primeiro crop dessa foto ficou em 5,219 s, contra 5,373 s sem prefetch na mesma rodada; a amostra não demonstra ganho robusto no processamento completo. O recurso permanece opcional para evitar I/O especulativo por padrão.

Com monitor desligado, a mesma rodada automática ficou em 2,074 / 5,169 / 2,014 s; ligado, 2,032 / 5,373 / 2,014 s. Não se atribui a diferença de aproximadamente 0,2 s da segunda foto exclusivamente à consulta de hardware: o rodapé altera a área disponível e há variação entre execuções. Os custos diretos do monitor foram instrumentados separadamente abaixo.

## Exportação, lote e fidelidade

Rodada final com duas repetições por foto/variante, amostragem CPU/RAM/GPU e o **preset automático efetivo**, sem fixar `OMP_NUM_THREADS` nem workers de staging. Medianas em segundos:

| Foto | Preset anterior, 28 threads | Novo automático | Redução |
|---|---:|---:|---:|
| 0950, sem denoise | 4,826 | **3,541** | 26,6% |
| 1012, denoise normal | 9,908 | **8,421** | 15,0% |
| 1020, denoise agressivo e cantos | 11,072 | **9,633** | 13,0% |

A ablação de decode/SIMD no início do documento tem três repetições e não usa amostragem GPU; não misturar as duas rodadas como se fossem uma só. No lote CLI com as três fotos no mesmo processo, uma execução por variante levou **24,979 → 21,657 s**, redução de **13,3%**. É um ensaio de lote sequencial, não de múltiplas exportações simultâneas nem da fila GUI em uso interativo.

Picos observados de RAM, controle/automático: 1998/2005 MiB na 0950, 5160/5158 MiB na 1012 e 6229/6113 MiB na 1020. A VRAM do processo chegou a 790 MiB nas duas primeiras e 1022 MiB na terceira, igual entre variantes. Utilização GPU é do dispositivo inteiro; as amostras não permitem atribuir toda a atividade ao ART. O backend CUDA é o da etapa anterior, com intermediários reutilizados; não foi transferido o pipeline inteiro para a GPU.

`ART_RAW_HASH=1`, em execuções separadas dos benchmarks, produziu **os mesmos dois hashes dos planos decodificados em cada uma das três fotos**, controle versus automático. São planos de 6740 × 4500 antes do recorte útil de 6732 × 4492. O teste SIMD independente também compara exatamente as amostras de saída.

Comparação dos TIFFs de 16 bits, ignorando contêiner e metadados:

| Foto | Erro médio entre duas execuções do controle, LSB | Controle × automático, LSB |
|---|---:|---:|
| 0950 | 0,00001247 | 0,00001195–0,00001238 |
| 1012 | 0,005236 | 0,005741–0,005857 |
| 1020 | 0,813602 | 0,816414–0,824995 |

A ordem dos desvios permanece próxima à variabilidade do próprio motor. Não há equivalência bit a bit do processamento completo: na 1020, o máximo pontual entre variantes chegou a 4232 LSB, e duas execuções do próprio automático chegaram a 4096 LSB. O comportamento não determinístico do denoise já documentado na etapa anterior continua uma limitação; esta validação não cobre outros sensores, X-Trans ou todos os perfis.

## Abertura progressiva e cache

- `ART_ASYNC_OPEN=1`: transfere `InitialImage::load` do modo editor simples para o pool; mantém a janela existente viva durante a decodificação. A abertura pelo navegador já usava uma tarefa assíncrona e passa a ter prioridade máxima.
- `ART_PROGRESSIVE_OPEN=1`: prepara em paralelo o JPEG embutido da câmera, limitado a 1280 × 854, e apresenta uma sobreposição com a legenda **“Prévia da câmera (sem ajustes) — processando RAW…”**. Não executa demosaic como fallback se faltar preview.
- A prévia permite reconhecer/compor a fotografia enquanto se espera. **Ainda não permite editar o RAW**, não aplica o perfil `.arp` nem a gestão de cor do editor. O processamento final e a apresentação com gestão de cor continuam no caminho normal.
- `ART_PREVIEW_CACHE=1`: LRU de até quatro prévias e 24 MiB em RAM, chave por caminho, tamanho e mtime com nanossegundos em Linux. Reabrir a foto usa a prévia em cache; o RAW continua sendo carregado. Alterar o `.arp` não invalida esse JPEG, pois ele explicitamente não contém ajustes.
- `ART_PREFETCH_PREVIEW=1`: experimento separado, desativado no preset; carrega somente o JPEG da próxima foto de mesma extensão, por ordem de nome. Trabalha em prioridade mínima. Não faz pré-decodificação RAW, não lê a pasta inteira em memória nem ocupa centenas de MiB por foto.
- Um contador de geração descarta solicitações de preview obsoletas. Trabalho já iniciado pode terminar, mas não substitui a foto atual. Não foi introduzida interrupção insegura de LibRaw ou dos filtros; no modo de editor único, a regra existente de não trocar durante processamento continua valendo.

Na triagem do lançamento direto `-s foto`, usando a 1020 e limite explícito de 28 threads, antes de o navegador existir, houve primeira prévia da câmera em aproximadamente 0,44 s **após o pedido interno de abertura**. A primeira foto processada passou de cerca de 5,49 para 4,06 s. O tempo de criação dos widgets ainda produz uma pausa inicial de aproximadamente 1,7 s na variante assíncrona. Não confundir esses tempos com a abertura a partir do navegador já pronto.

## CPU, filas e dependências

`rtengine/cpupolicy` detecta a máscara de CPUs permitidas, núcleos físicos distintos, irmãos SMT e sockets. Em Linux consulta também nós NUMA, capacidade publicada em `cpu_capacity`, linha de cache, maior cache informado e limites visíveis `cpu.max` de cgroup v2, inclusive ancestrais. O cálculo de quota segue o [formato documentado pelo kernel](https://docs.kernel.org/admin-guide/cgroup-v2.html). Cgroups v1 e montagens v2 fora de `/sys/fs/cgroup` não são interpretados. Topologia indisponível usa uma estimativa conservadora a partir dos processadores lógicos.

A política `dynamic` usa núcleos físicos disponíveis como ponto de partida, deixa um núcleo de margem quando há mais de dois e respeita um `OMP_NUM_THREADS` explícito, limitado aos recursos detectados. Não fixa afinidade. O pool reserva capacidade para aceitar trabalho interativo e limita tarefas de fundo a uma enquanto há abertura/preview/crop ativo; fora disso permite até `min(8, físicos/4)`. Uma tarefa de fundo usa uma thread OpenMP. Tarefas que já estavam executando terminam normalmente; a limitação vale para novas admissões.

A política experimental `groups` separa um núcleo físico completo para UI, 2–8 para tarefas auxiliares e os demais para processamento, incluindo os irmãos SMT no mesmo grupo. Só cria grupos em máquinas com pelo menos oito núcleos físicos. Não presume que a CPU lógica 0 esteja permitida. O OpenMP pode manter equipes já criadas; esta política não constitui um gerenciador completo de afinidade de todas as threads da aplicação. Não foi escolhida para o preset.

O modelo conserva as dependências do motor: JPEG e RAW independentes; editor vinculado somente depois do RAW; publicação visual pelo loop GTK; saída de filtros dependente dos seus dados de entrada. Reutiliza a fila central de prioridades e o paralelismo CPU/GPU do estágio anterior. **Não foi implantado work stealing**: os gargalos medidos eram o mutex de decode e o processamento, não contenção demonstrada da fila. Mais pools irrestritos criariam equipes OpenMP concorrentes e pressão de memória.

Não há migração automática entre P/E cores, pinagem de memória NUMA ou calibração por tamanho de LLC. O sistema testado tem dois sockets, mas expõe apenas um nó NUMA; a detecção não equivale a validação em uma máquina heterogênea ou multi-NUMA. Exportações e processamento de edição ainda podem competir por equipes de CPU; não se implementou um orçamento global de todos os filtros em execução.

## SIMD e portabilidade

A conversão de quatro `uint16_t` intercalados por pixel para o plano Bayer ganhou um caminho AVX2 opcional, com gather, seleção de canal, conversão exata para float e cauda escalar. O despacho usa detecção em tempo de execução; o atributo `target("avx2")` fica restrito à função em GCC/Clang x86. Demais compiladores/arquiteturas mantêm o loop comum. O caminho é acionado somente por `ART_RAW_SIMD=1`, que **fica desligado no preset**.

A política registra SSE2, SSE4.1/4.2, AVX, AVX2, FMA e AVX-512F quando a plataforma oferece a consulta. Isso não significa que todos possuam novos kernels. SSE já existe em várias rotinas do ART, inclusive filtros recursivos e conversões. FMA não foi introduzido nas referências numéricas: pode mudar o arredondamento. A CPU usada não tem AVX-512. MMX não resolve um gargalo identificado.

Em 30,24 MP e 28 threads, o loop comum compilado levou aproximadamente 6,37 ms e AVX2 6,47 ms nas iterações aquecidas. O volume lido e escrito é cerca de 346 MiB; os tempos são compatíveis com forte limitação por memória. Na exportação completa, AVX2 sozinho passou de 4,774 a 4,724 s, diferença pequena demais para justificar habilitação automática.

Na varredura adicional de 1/4/28 threads, medianas das duas passagens aquecidas foram aproximadamente **42,30/13,35/6,82 ms** no loop comum e **39,37/11,25/6,85 ms** no AVX2. Há ganho isolado com poucas threads, mas não no ponto de operação multicore utilizado. O diagnóstico `-fopt-info-vec-all` do GCC confirmou que o loop comum dessa função não foi autovetorizado; ainda assim, intrinsics não resolveram o limite multicore. Conversões em `color.cc` e gaussianos já possuem caminhos SSE2; ampliar a ISA sem medir o estágio dominante não foi adotado.

O teste independente compila com `-O3`, **sem `-march=native`**, e verifica exatamente todos os 16 pares de canais, larguras de 1 a 6732, caudas e limites. Os binários principais locais continuam sendo builds `native` para medição. Para distribuição portátil deve-se usar `PROC_TARGET_NUMBER=1`: uma função com despacho dinâmico não torna portátil um executável inteiro compilado com `-march=native`, conforme a [documentação do GCC](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html).

## Monitor integrado

O monitor ocupa o espaço livre da barra superior existente do editor e mostra CPU total, GPU, RAM do processo, VRAM e tarefas auxiliares ativas/em fila, sem acrescentar uma faixa ao layout. Quando falta espaço, o texto é abreviado com reticências; os valores completos aparecem ao passar o mouse. Os benchmarks acima foram feitos antes dessa mudança visual, com o monitor ainda no rodapé. A consulta ocorre em thread própria a cada segundo; o callback GTK apenas lê o texto pronto e os contadores. As mensagens estão no sistema de traduções do ART. `ART_HW_MONITOR=0` oculta o monitor.

Em Linux, CPU e RSS usam `/proc`; GPU NVIDIA usa NVML carregado opcionalmente com `dlopen`, sem dependência de headers CUDA/NVML para compilar. GPU e VRAM referem-se **ao dispositivo 0 inteiro**, incluindo desktop e outros processos; o tooltip explica isso. A API é descrita na [documentação NVML](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html). Ausência de driver/API ou plataforma não implementada produz “—”, sem impedir o uso do ART. O monitor não seleciona GPUs AMD/Intel nem acompanha automaticamente outra GPU escolhida pelo CUDA.

No ensaio instrumentado com NVML real, a consulta periódica levou **229 µs de mediana / 1768 µs de máximo** em thread auxiliar; o callback GTK, **31 / 63 µs**. A inicialização NVML levou 17,88 ms, também fora da UI. Esses são custos diretos observados em uma execução, não uma garantia de overhead zero em toda máquina.

O contador representa o pool auxiliar, não todos os workers OpenMP, threads CUDA e tarefas internas de crop. Não interpretar zero nesse contador como ausência de processamento. O monitor está ligado por padrão; as novas políticas de processamento continuam experimentais e dependem das flags/preset.

## Correções de estabilidade

- Cada abertura tem seu próprio código de erro; o ponteiro compartilhado entre carregamentos concorrentes foi removido.
- `ProgressConnector` captura exceções e publica a conclusão somente depois de liberar seu mutex; o callback pode destruir o conector sem o worker voltar a acessar o objeto.
- Contadores de jobs/miniaturas foram sincronizados; a conclusão agora depende de fila vazia e nenhum trabalho ativo, com decremento e verificação sob o mesmo lock.
- Preview obsoleto não é publicado; falha de RAW remove a sobreposição; o caminho assíncrono libera a referência temporária da miniatura e protege o fechamento durante o carregamento.

## Reprodução e configuração

O script passa a detectar recursos em tempo de execução; não exporta mais `OMP_NUM_THREADS=28` nem fixa quatro workers de staging. Variáveis explícitas do usuário prevalecem. Para testar nesta sessão:

```sh
LD_LIBRARY_PATH=/tmp/art-build-deps/usr/lib/x86_64-linux-gnu \
  bash tools/run_art_optimized.sh /tmp/art-build-cpu/rtgui/ART -N
```

`-N` cria uma instância nova, importante para não encaminhar a abertura para um ART já iniciado com outras flags. O script exige o executável como argumento. Não instala binários nem altera configurações persistentes. `ART_CUDA=0` mantém o preset CPU. Para comparar com o estágio anterior, definir `ART_SCHEDULER=legacy ART_RAW_PARALLEL=0 ART_ASYNC_OPEN=0 ART_PROGRESSIVE_OPEN=0 ART_PREVIEW_CACHE=0 ART_HW_MONITOR=0 OMP_NUM_THREADS=28 ART_HYBRID_WORKERS=4` além das flags antigas.

```sh
cmake -S tools/performance-tests -B /tmp/art-performance-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/art-performance-tests -j4
ctest --test-dir /tmp/art-performance-tests --output-on-failure

# Matriz e ações estão nos dados desta etapa; grave-os em arquivos JSON.
# null em uma variável da matriz remove a variável herdada do ambiente.
dbus-run-session -- python3 tools/probe_art_navigation.py \
  --cli /caminho/ART --matrix matriz.json --actions acoes.json \
  --output /tmp/navegacao-nova --repeat 2 \
  /caminho/276A0950.cr2 /caminho/276A1012.cr2 /caminho/276A1020.cr2

python3 tools/benchmark_art.py --cli /caminho/ART-cli \
  --matrix matriz-exportacao.json --output /tmp/exportacao-nova \
  --repeat 3 /caminho/276A0950.cr2
```

O harness de GUI exige Xvfb, libXtst e ImageMagick. As coordenadas das ações foram verificadas nas capturas desta configuração; outras escalas/layouts exigem inspeção e ajuste. Por padrão termina ART após o ensaio; `--close-at-end` envia `WM_DELETE_WINDOW` e exige saída normal com código zero. Essa opção foi usada nas confirmações e nos testes de fallback. `ART_OPEN_TRACE=1` e `ART_UI_PROFILE=1` produzem timestamps monotônicos; `ART_PROFILE=1` produz durações inclusivas — não somar pai e filhos. `ART_RAW_HASH=1` acrescenta um hash FNV dos planos decodificados para diagnóstico e deve permanecer desligado nos benchmarks.

## Verificação final e gargalos restantes

- GUI e CLI compilados com e sem CUDA. Os dois testes CTest de política/conversão passaram; mais dez execuções sob `taskset` cobriram 1, 2, 4 e 8 CPUs e dois irmãos SMT, em ambas as políticas. Dois lógicos do mesmo núcleo foram corretamente contados como um físico. Restringir afinidade não simula todos os efeitos de uma CPU pequena real.
- Quatorze sessões de navegação finais, 42 aberturas, com hashes de sidecars verificados. A confirmação controle/automático encerrou normalmente, com código zero.
- Duas sessões adicionais sem CUDA verificaram fechamento solicitado durante decode (adiado enquanto ocupado) e fechamento normal após o editor ficar pronto. Uma delas injetou uma biblioteca NVML que retorna indisponibilidade: GPU/VRAM apareceram como “—”, e a edição continuou funcionando.
- A pausa de construção inicial dos widgets, eventos longos da UI e a variabilidade do denoise permanecem. A política dinâmica não demonstrou melhora uniforme de p95/pior latência.
- `histmatching.cc::getAutoMatchedToneCurve` ainda chama `Thumbnail::loadFromRaw` para gerar a referência de ajuste automático, podendo decodificar novamente uma foto já carregada. Reutilizar os dados exigiria preservar o processamento neutro e suas condições de cor; não foi substituído silenciosamente por uma aproximação.
- Prévia em RAM não é cache RAW em VRAM. Os testes anteriores de exibição, resize, OpenCL e Tensor Cores continuam descritos nos relatórios vinculados; esta etapa não muda sua conclusão nem introduz kernels com precisão reduzida.
- Os novos controles são variáveis de ambiente, ainda sem uma página de preferências. Não houve validação em hardware AMD/Intel GPU, P/E cores, múltiplos nós NUMA ou outros sistemas operacionais.


## Revisão adicional de estabilidade e desempenho — 26/10/2026

A revisão concentrou-se em esperas de miniaturas, descarte de imagens, leitura RAW multiframe e tarefas em segundo plano. As correções estão no código; esta rodada não estabelece um novo ganho percentual de velocidade.

- **Espera de miniaturas:** a conclusão podia ocorrer entre a leitura do contador e a entrada na espera, perdendo a notificação e bloqueando a interface. A fila e o predicado de espera agora compartilham o mesmo mutex. A conclusão usa RAII, inclusive quando o processamento lança exceção; remover uma entrada espera somente os trabalhos daquela entrada.
- **Cancelamento e memória:** o descarte de callbacks GTK podia deixar a imagem e seu auxiliar sem liberação. O callback agora possui a imagem e a libera também ao ser cancelado. A destruição da entrada aguarda seus produtores antes de cancelar os callbacks.
- **RAW multiframe:** uma falha de um quadro podia ser sobrescrita pelo sucesso do quadro seguinte na mesma thread. Erros e exceções agora são guardados por quadro e propagados após a conclusão da região paralela. O número de workers não ultrapassa o número de quadros. A seleção de somente um quadro considera a contagem real do arquivo, preservando a regra de ignorar o composto inicial em arquivos especiais com sete ou mais quadros. Quadros e buffers parcialmente carregados são liberados mesmo quando o número de quadros utilizáveis diminui; a fonte da imagem tem propriedade RAII durante o carregamento.
- **Monitor:** continua no espaço livre da barra superior existente. O worker só começa quando o editor aparece; a coleta e o timer da interface ficam suspensos enquanto ele está oculto. O texto só é atualizado quando muda.
- **Instâncias secundárias e CUDA:** o aquecimento assíncrono ocorria também no processo usado apenas para encaminhar uma pasta à janela existente. Os ensaios reproduziram timeout e aborto no processo secundário, com inicialização CUDA concorrendo com a saída. O aquecimento agora começa somente quando a GUI cria sua janela; a CLI mantém o aquecimento antecipado. Um guard em `main` aguarda sua conclusão antes da destruição dos objetos estáticos, incluindo saídas por erro.

### Validação desta revisão

- GUI e CLI compiladas com CUDA e sem CUDA.
- Quatro testes CTest passaram: políticas de CPU, seleção/erros/exceções dos quadros e espera de miniaturas. Os novos casos rodam com e sem OpenMP e cobrem limites de 1, 2 e 28 threads, conclusão antes da espera, espera por entrada e liberação após exceção.
- O teste isolado passou com AddressSanitizer e UndefinedBehaviorSanitizer. Isso não equivale a executar toda a aplicação com sanitizadores.
- Na GUI isolada com Xvfb, o monitor produziu zero amostras antes de abrir o editor. Ao ocultá-lo, o contador permaneceu em 4 durante a janela medida de três segundos; ao voltar, chegou a 7. Antes da correção, continuava coletando enquanto oculto.
- Depois da correção de CUDA, 12 encaminhamentos alternados entre duas pastas encerraram com código zero, as miniaturas reapareceram e a janela principal fechou normalmente. As fotos e perfis usados na GUI eram cópias.
- Comparação antes/depois com `276A0950`, `276A1012` e `276A1020`: os hashes dos seis quadros RAW decodificados permaneceram iguais. As exportações TIFF continuam sujeitas à variabilidade do denoise já documentada; os erros observados estão registrados no JSON, sem promessa de igualdade bit a bit do TIFF.
- Um arquivo `.cr2` inválido foi recusado, sem gerar TIFF, sem travamento ou aborto e sem erro de encerramento do driver CUDA.

Dados: [performance-review-20261026.json](benchmarks/performance-review-20261026.json). Os tempos com `ART_RAW_HASH=1` são diagnósticos, não benchmarks. A seleção especial de quadros foi coberta por testes sintéticos; não havia um RAW Hasselblad multishot para validação real. O recarregamento para histogram matching e os limites de latência descritos acima permanecem como pontos de investigação.
