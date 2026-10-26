# CUDA seletivo para deconvolução

Implementação experimental de 26/10/2026 sobre ART `4a7ed4fb775f555d54787a2748d923ed5c9a9223`.

**Evolução posterior:** este documento registra a primeira etapa. A [etapa avançada](advanced-performance.pt-BR.md) acrescenta Gaussiana recursiva opcional, staging pinned, par CPU/GPU e dois streams; seus limites e orçamento de memória estendem os descritos abaixo.

## Escopo

O primeiro bloco acelerado é a deconvolução Richardson–Lucy da ferramenta de nitidez (`Method=rld`), nas variantes Gaussianas diretas de 3×3, 5×5 e 7×7. As vinte iterações mantêm intermediários na GPU. Preparação, upload, download, validação e publicação do resultado fazem parte do custo medido. O restante do pipeline continua na CPU.

Este é um port da matemática existente no ART, sem incorporar código do LightCraft. Foram aproveitadas as ideias de residência de intermediários, reutilização de buffers, CPU como referência e fallback. Não usa Tensor Cores.

O build padrão permanece sem dependência CUDA. Mesmo em um build CUDA, é necessário `ART_CUDA=1` para ativar o protótipo. Ele não representa uma seleção automática calibrada para todas as placas.

## Compilar e executar

Com as dependências normais de compilação do ART instaladas:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DART_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target art-cli -j
```

`86` corresponde à A4500 usada nos testes; selecionar arquiteturas adequadas às GPUs de destino. CUDA requer CMake ≥ 3.18 e toolkit compatível com o compilador host. O caminho CPU mantém o requisito anterior de CMake. Instalar/preparar os dados do ART como em um build normal antes de executar o CLI.

```sh
ART_CUDA=1 build-cuda/rtgui/ART-cli -a -o /tmp/resultado.tif \
  -t -b16 -p /caminho/foto.cr2.arp -c /caminho/foto.cr2
```

`ART_CUDA=0`, ou variável ausente, mantém o processamento CPU. `-DART_ENABLE_CUDA=OFF` remove o backend do build. A mesma implementação é chamada pelo motor compartilhado da interface, mas os testes desta etapa foram feitos no CLI, não na interação gráfica.

## Elegibilidade e fallback

- Sigma efetivo (`deconvradius / scale`) entre **0,25 e 1,15**, inclusive. Raios maiores usam a Gaussiana recursiva original na CPU. O raio automático pode diferir daquele salvo no perfil.
- Imagem entre **1.048.576 e 67.108.864 pixels**, dimensões mínimas de 8 pixels e amount positivo/finito. Esses limites delimitam o protótipo; não são limiares universais de desempenho.
- Entradas não finitas, GPU indisponível, contexto ainda em preparação, outra chamada CUDA em andamento ou erros de recursos/execução retornam ao caminho CPU.
- O contexto é preparado em uma thread durante a inicialização do motor. A thread é aguardada ao encerrar. Se a preparação ainda estiver em andamento, o filtro não espera por ela.
- O resultado só altera a imagem host depois de download completo, sincronização e verificação de finitude. Uma falha CUDA desativa esse backend durante o restante do processo, com motivo em stderr.
- Um conjunto de buffers reutilizáveis é mantido até encerrar o processo. No máximo, são cinco planos FP32 e um plano de bytes no dispositivo (21 bytes/pixel), mais dois planos FP32 e um de bytes no host (9 bytes/pixel), sem contar os buffers normais do ART. Para 30,24 MP: aproximadamente 606 MiB no dispositivo e 260 MiB no host. A memória acompanha o maior pedido aceito; não cresce sem limite por foto.

Não há GPU para denoise, demosaic, Gaussiana recursiva ou deconvolução com PSF arbitrária nesta etapa. Não há mudança dos controles ou dos perfis `.arp`.

## Instrumentação reproduzível

`ART_PROFILE=1` habilita linhas JSON prefixadas por `ART_PROFILE` em stderr, sem depender de `Verbose` ou `WITH_BENCHMARK`. Os tempos usam relógio monotônico e incluem dimensões e pipeline. Escopos são inclusivos: não somar `processImage`, seus estágios e operações filhos.

`cuda_deconvolution_attempt` inclui todo o custo do bloco, inclusive alocação quando necessária. `cuda_prepare_buffers`, `cuda_pack`, `cuda_transfer_and_compute` e `cuda_commit` permitem identificar overhead. O tempo de inicialização `cuda_context_init` ocorre em segundo plano e não deve ser somado ao processamento como se fosse serial. Uma tentativa sem êxito não prova que a imagem usou GPU.

O utilitário abaixo usa os sidecars existentes apenas para leitura, cria configurações/cache isolados e exporta TIFFs de 16 bits para uma pasta nova. Ele registra comando, logs, eventos e duração do processo em `results.json`:

```sh
ART_CUDA=0 python3 tools/profile_pipeline.py \
  --cli /caminho/ART-cli --output-dir /tmp/medicao-cpu \
  --threads 28 --repeat 3 /caminho/foto.cr2
ART_CUDA=1 python3 tools/profile_pipeline.py \
  --cli /caminho/ART-cli --output-dir /tmp/medicao-gpu \
  --threads 28 --repeat 3 /caminho/foto.cr2
```

O diretório de saída deve ser inexistente. Usar `--profile` para fornecer um perfil comum, se necessário. Não executar benchmarks junto de compilações ou outras cargas intensivas. Cada exportação do utilitário inicia um novo processo; portanto, inclui startup e primeira alocação, e não simula o cache quente da interface. O tempo total também inclui inicialização do aplicativo, leitura, codificação e escrita, além do motor.

## Verificação de qualidade

`ART_CUDA_VERIFY=1 ART_CUDA=1` executa o bloco nas duas implementações sobre a **mesma entrada**, imprime `ART_CUDA_VERIFY` com sigma efetivo, quantidade de amostras diferentes, não finitas e erros absoluto máximo/médio. Nesse modo, a imagem retornada é a da CPU. Usar para diagnóstico de qualidade, nunca para comparar desempenho.

Os testes isolados compilam a implementação CPU real de `rtengine/deconvolution.cc` e `gauss.cc`, sem o despacho CUDA. Cobrem campos constantes, ruído, bordas, valores negativos e acima de 65535, máscaras e pixels de impulso, padding de linhas e fronteiras entre os três kernels. Também verificam retorno intacto em entradas não suportadas e ausência de dispositivo. Uma GPU ausente faz o teste de equivalência falhar; não é tratada como aprovação silenciosa.

```sh
cmake -S tools/cuda-tests -B build-cuda-tests -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-tests -j
ctest --test-dir build-cuda-tests --output-on-failure
ART_CUDA=1 OMP_NUM_THREADS=28 build-cuda-tests/deconvolution_test 6732 4492
build-cuda-tests/compare_tiff /tmp/cpu.tif /tmp/gpu.tif
```

`compare_tiff` compara amostras decodificadas de TIFF unsigned de 16 bits, sem confundir diferenças de metadados com diferenças de pixels. Retorna 0 somente se forem idênticas, 1 se houver diferenças e 2 se não puder comparar. Não é um teste de equivalência perceptual.

## Ambiente e decisões

Máquina de teste: dois Xeon E5-2680 v4 (28 núcleos físicos/56 threads), RTX A4500, driver 615.71.09, CUDA 13.4, GCC 14.2, Release com `-march=native`, OpenMP e contração FMA desativada. Foram escolhidas 28 threads após observar que 56 pioravam a primeira amostra. CPU e GPU usam a mesma configuração.

As dependências que faltavam foram baixadas e extraídas em `/tmp/art-build-deps`, sem instalação no sistema. Os executáveis e dados de execução ficaram em `/tmp`. Mimalloc e OpenColorIO foram desativados nos builds comparados; os perfis selecionados para comparação não usam simulação de filme. Não foi feita instalação do ART nem alteração das fotos ou dos sidecars.

Os sidecars fornecidos declaram ART 1.26.9 / versão de perfil 1045, enquanto este fork declara `PPVERSION 1043`. As medições comparam CPU e GPU deste mesmo fork lendo esses perfis; não comprovam equivalência com a saída de outra versão do ART nem suporte a eventuais campos mais novos.

O port mantém inclusive a expressão assimétrica `c21` do kernel 7×7 existente em `gauss.cc`. Corrigi-la aqui alteraria o resultado dos perfis; essa possível correção exige uma investigação separada. `--fmad=false` no CUDA preserva a ordem das operações FP32.

A opção antiga `WITH_BENCHMARK=ON` falhava ao expandir `BENCHFUN` fora do namespace `rtengine`. O macro agora usa o nome qualificado da classe; isso permitiu compilar a referência instrumentada.

## Limites da conclusão

Esta etapa valida um bloco delimitado e uma GPU, não o motor inteiro em GPU nem todas as combinações de hardware e parâmetros. A decisão de habilitar por padrão depende de ampliar o corpus e validar a interface. O denoise tradicional continua sendo o principal candidato de maior impacto nas fotos com ruído; seu custo foi medido separadamente para orientar a próxima etapa.

## Resultados medidos

Foram inspecionados 77 CR2 e 36 sidecars da pasta fornecida pelo usuário. Entre os sidecars, 34 habilitam nitidez, 22 equalizador tonal e 8 denoise. Uma triagem em três fotos orientou a escolha de um port delimitado de nitidez. A comparação final usou duas dessas fotos, seus perfis originais, o mesmo executável em modos CPU/GPU e três processos novos por foto/modo. Nenhuma compilação ocorreu durante essas medições finais.

| Foto / trecho | Mediana CPU | Mediana CUDA | Observação |
|---|---:|---:|---|
| 276A0950 — deconvolução | 413,83 ms | 127,56 ms | 3,24×; todo o bloco, incluindo transferências e validação |
| 276A1012 — deconvolução | 423,98 ms | 122,54 ms | 3,46×; todo o bloco |
| 276A0950 — motor `processImage` | 1,788 s | 1,482 s | Escopo do motor; não inclui todo o processo CLI |
| 276A1012 — motor `processImage` | 10,208 s | 9,844 s | Denoise continua dominante |
| 276A0950 — processo CLI completo | 4,900 s | 4,776 s | Redução de 2,5% na mediana; intervalos se sobrepõem |
| 276A1012 — processo CLI completo | 12,951 s | 12,750 s | Redução de 1,6% na mediana; amostra pequena |

Intervalos min–max do processo completo: 276A0950 CPU **4,723–5,040 s**, GPU **4,672–4,784 s**; 276A1012 CPU **12,948–12,980 s**, GPU **12,675–12,752 s**. As medianas totais não sustentam uma promessa geral de aceleração da exportação; o resultado claro desta etapa está no bloco de deconvolução. Não são medidas de latência da interface.

A primeira versão CUDA era mais lenta na exportação isolada porque inicializava o contexto dentro do filtro. Separar a inicialização e paralelizar o preparo/validação dos buffers foi necessário para alcançar os resultados acima. No teste isolado de 30,24 MP, a primeira chamada completamente fria ainda custou **509 ms**, contra **457 ms** CPU. As chamadas seguintes ficaram em **112–119 ms** CUDA contra **400–464 ms** CPU, para diferentes sigmas/padrões. Essa varredura é uma caracterização, não uma distribuição estatística de um único caso.

Na foto 276A1012, o denoise consumiu **4,87 s** e o cálculo de seus parâmetros **2,01 s** na mediana CPU. Esses são candidatos mais relevantes para uma próxima investigação do que migrar indiscriminadamente curvas e exposição.

Verificações numéricas realizadas:

- Oito casos sintéticos em aproximadamente 1 MP e os mesmos oito em **6732 × 4492**: zero diferenças FP32 entre CPU e GPU nesta máquina.
- CTest: **2/2 testes aprovados**, incluindo dispositivo indisponível; NVIDIA Compute Sanitizer (`memcheck`): **zero erros**. Builds Release do CLI com CUDA ligado e desligado concluídos; o executável CPU não liga a bibliotecas CUDA.
- Modo de comparação sobre a mesma entrada real: zero diferenças e zero valores não finitos em ambas as fotos. Sigmas efetivos: aproximadamente **0,933073819** e **0,348989695**.
- Exportação completa de 276A0950 com uma thread CPU: TIFFs de 16 bits CPU/GPU idênticos, assim como duas execuções CPU de controle.
- Com 28 threads, há pequenas diferenças já entre exportações CPU/CPU. Por exemplo, 276A1012 apresentou cerca de 65–67 mil amostras diferentes em 13,1 milhões, erro médio próximo de 0,0053 LSB e máximo 112 LSB; também há variação entre GPU/GPU. Por isso, a prova do port usa a mesma entrada do filtro, sem atribuir toda variação do pipeline ao CUDA.

Logs e imagens desta sessão estão em diretórios temporários: `/tmp/art-measured-cpu`, `/tmp/art-measured-gpu`, `/tmp/art-cuda-large-optimized.log`, `/tmp/art-quality-cpu`, `/tmp/art-quality-gpu` e `/tmp/art-verify-optimized`. Eles não fazem parte do repositório; os comandos acima permitem reproduzir a medição com outros arquivos.
