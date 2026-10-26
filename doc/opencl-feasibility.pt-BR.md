# OpenCL no ART: comparação medida com CPU e CUDA

Teste de 26/10/2026 na **NVIDIA RTX A4500**, usando os mesmos algoritmos e a mesma precisão dos experimentos anteriores. Dados completos, incluindo as duas rodadas, tempos iniciais e intervalos: [JSON das medições](benchmarks/opencl-comparison-20261026.json).

## Conclusão para acelerar o uso do ART

**OpenCL funcionou e acelerou a deconvolução em relação à CPU, mas não mostrou vantagem consistente sobre CUDA nesta GPU.** A API não resolveu o custo de transportar a imagem entre RAM e VRAM. Para esta máquina, não há evidência para substituir o backend CUDA existente por OpenCL visando desempenho.

O experimento está implementado em `tools/opencl-tests`, com deconvolução, Lanczos, cache explícito da origem, referência CPU e comparação opcional com CUDA. **Não foi conectado ao editor nem à exportação do ART.** As opções atuais de execução do programa continuam iguais; não existe um novo `ART_OPENCL=1` que ative processamento no aplicativo.

O caminho com maior potencial para o uso habitual continua sendo acelerar os blocos caros ainda executados na CPU. Na [medição anterior da foto 276A1012](cuda-deconvolution.pt-BR.md#resultados-medidos), o denoise levou 4,87 s e o cálculo de seus parâmetros 2,01 s. São oportunidades de maior impacto que trocar a API de um bloco de deconvolução que já leva aproximadamente 0,12 s na GPU. Esses tempos de denoise são da investigação anterior, não de uma implementação OpenCL dessa operação.

## Ambiente confirmado

- Driver NVIDIA **615.71.09**; dispositivo selecionado: **NVIDIA RTX A4500**.
- Versão anunciada pelo driver: `OpenCL 3.0 CUDA`; linguagem do kernel: `OpenCL C 1.2`.
- API host e kernels usam o subconjunto **OpenCL 1.2**, suficiente para estas operações. O nome anunciado pelo driver não significa que o experimento chamou o backend CUDA do ART: os kernels OpenCL foram compilados e executados pela API OpenCL.
- CPU: dois Xeon E5-2680 v4, com `OMP_NUM_THREADS=28`. Referências CPU e host sem contração FMA; CUDA 13.4 com `--fmad=false`.
- Apenas GPUs OpenCL disponíveis e com compilador são elegíveis; não há seleção silenciosa de um dispositivo OpenCL CPU.
- Os cabeçalhos de desenvolvimento ausentes foram extraídos em `/tmp/art-opencl-deps`, sem instalação de pacotes no sistema.
- O build OpenCL independente não habilita a linguagem CUDA, não usa `nvcc` e não vincula `cudart`. Foi executado com o loader do sistema `/usr/lib/x86_64-linux-gnu/libOpenCL.so.1`, além da rodada inicial com o loader distribuído pelo toolkit. O driver OpenCL continua necessário.

Somente esta GPU/driver foi validada. Usar OpenCL permite investigar outros fabricantes, mas os resultados não comprovam funcionamento ou ganho em AMD/Intel.

## Comparação na foto de 30 MP

Entrada de **6732 × 4492**: TIFF RGB16 previamente exportado de `276A0950.cr2`, da pasta fornecida pelo usuário. RAW e perfil originais preservados. A deconvolução usa o canal verde convertido para FP32 e máscaras determinísticas de blend/impulso; o resize usa os três planos FP32. Isso testa a operação sobre pixels da foto, **não é uma captura dos intermediários Lab do editor**.

CPU, OpenCL e CUDA foram executados no mesmo processo. A ordem é alternada a cada repetição; seis amostras após uma primeira chamada por caso. Sem compilação ou exportação concorrente. A tabela usa a rodada final com o loader OpenCL do sistema.

| Operação | CPU | CUDA | OpenCL |
| --- | ---: | ---: | ---: |
| Deconvolução, sigma 0,34899, 20 iterações | 414,8 ms | 110,0 ms | 114,0 ms |
| Deconvolução, sigma 0,93307, 20 iterações | 428,0 ms | 121,7 ms | 121,1 ms |
| Lanczos → 2560 × 1708, nova origem incluindo upload | 34,0 ms | 112,9 ms | 116,0 ms |
| Lanczos → 2560 × 1708, origem já residente | 34,0 ms | 27,4 ms | 28,1 ms |
| Lanczos → 1683 × 1123, nova origem incluindo upload | 35,3 ms | 93,5 ms | 95,1 ms |
| Lanczos → 1683 × 1123, origem já residente | 35,3 ms | 6,4 ms | 6,4 ms |

São tempos totais da operação, incluindo preparação, alocações necessárias, transferências, execução, verificação de finitude e publicação em RAM. A coluna residente exclui somente o upload da origem; **o download da saída continua incluído**. Decodificação do TIFF, preparação da referência e comparação CPU/GPU estão fora do cronômetro. Não se mediu exportação completa nem interação na interface nesta etapa.

OpenCL acelerou a deconvolução aproximadamente **3,5–3,6×** frente à CPU. As diferenças para CUDA são pequenas e os intervalos se sobrepõem. Na outra rodada, as medianas OpenCL/CUDA foram 111/112 ms e 117/113 ms nos dois sigmas: a conclusão permanece, sem atribuir pequenas diferenças exclusivamente à API ou ao loader.

Na saída de 2560 pixels, a vantagem do cache sobre a CPU foi modesta nesta rodada. Na etapa anterior, CPU e GPU residentes chegaram a ficar empatadas ou com a CPU à frente; portanto, esse caso ainda não justifica habilitação automática. Na redução para 1683 pixels, o cache trouxe ganho claro em ambas as APIs, desde que a mesma origem seja reutilizada.

### Inicialização e transferências

A primeira deconvolução OpenCL do processo levou **463,5 ms**, contra 353,6 ms no primeiro uso CUDA e 443,1 ms na chamada CPU daquela rodada. Inclui criação de contexto, compilação/carregamento do programa e alocação inicial, além do filtro. O cache de compilação do driver já podia conter resultados dos testes anteriores; **não é uma medição de compilação com cache vazio**. A rodada anterior registrou 639 ms no primeiro uso OpenCL. A inicialização não deve bloquear a interação do editor numa integração futura.

O preparo/upload da origem do resize levou aproximadamente **88–90 ms** em OpenCL. Trocar CUDA por OpenCL não eliminou essa transferência. Copiar uma foto nova para executar apenas esse resize continuou perdendo para a CPU.

## Implementação e qualidade

Os kernels foram portados dos experimentos CUDA/algoritmos CPU já presentes neste trabalho, sem incorporar um filtro diferente. Deconvolução preserva suporte gaussiano, bordas, offset, máscaras, parada e vinte iterações; o Lanczos preserva os coeficientes calculados na CPU e a ordem vertical → horizontal.

`#pragma OPENCL FP_CONTRACT OFF` e `-cl-fp32-correctly-rounded-divide-sqrt` preservam o contrato numérico escolhido. Não se usaram `fast-relaxed-math`, FP16 ou Tensor Cores. O protótipo consulta a capacidade de divisão corretamente arredondada e rejeita dispositivos sem ela. As opções de compilação são passadas por [`clBuildProgram`](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clBuildProgram.html); a fila é criada para executar comandos em ordem, conforme a [API OpenCL](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clCreateCommandQueue.html).

Resultados:

- **4/4 testes OpenCL passaram**, incluindo ausência de plataforma via diretório ICD vazio, sem simular sucesso em CPU.
- **4/4 testes CUDA passaram novamente**, depois de compartilhar a mesma bateria de testes entre os backends.
- Campos constantes, ruído, negativos, bordas fortes, oito sigmas com transições 3×3/5×5/7×7, redução/ampliação e linhas com padding foram comparados com a CPU.
- **Zero diferenças FP32** nos testes executados e nas duas rodadas com a foto real. Isso não é garantia para todas as GPUs ou entradas.
- Troca da origem, alteração de dimensões, entradas não finitas, limites e falhas sem alteração da saída foram exercitados. Não foi executado um sanitizador de memória de dispositivo OpenCL; os resultados de Compute Sanitizer da etapa CUDA não cobrem esses kernels.

A deconvolução reutiliza cinco planos FP32 e um de bytes, limitados a 64 Mi pixels: cerca de **606 MiB** de buffers GPU para esta foto. O cache de resize guarda uma origem decodificada e possui limite de **1 GiB por instância**. Aqui, origem + trabalho ocupam aproximadamente 528 MiB ou 454,5 MiB, conforme a saída. `upload` invalida a imagem anterior mesmo se falhar; `clear` libera as referências aos buffers. O protótipo compartilha contexto/fila e pertence a uma única thread chamadora; ainda exigiria integração com concorrência e ciclo de vida do editor.

## Relação com exibição e prioridades

Este teste OpenCL é de **cálculo**, não substitui o teste de apresentação OpenGL/GTK. Compartilhar memória OpenCL/OpenGL é uma possibilidade prevista na [especificação Khronos](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html#cl_khr_gl_sharing), com sincronização e contextos compatíveis; não foi implementado neste protótipo. Não se deve supor que isso elimine as cópias exigidas pelas etapas que continuam na CPU.

Para tornar o ART mais rápido nesta máquina, a evidência favorece:

1. Manter o backend CUDA seletivo existente para os casos já validados.
2. Concentrar novos ports no denoise e em blocos caros que possam reutilizar intermediários na GPU.
3. Usar cache de pixels processados quando houver reutilização suficiente; decidir por medições que incluam transferências e invalidação.
4. Considerar OpenCL como alternativa para ampliar cobertura de hardware, com testes nos dispositivos-alvo, sem esperar ganho automático na A4500.

## Reproduzir

Somente OpenCL, sem toolkit CUDA; requer headers/loader OpenCL, OpenMP, TIFF, glibmm e lcms2 de desenvolvimento:

```sh
cmake -S tools/opencl-tests -B /tmp/art-opencl-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/art-opencl-tests -j
ctest --test-dir /tmp/art-opencl-tests --output-on-failure
OMP_NUM_THREADS=28 /tmp/art-opencl-tests/gpu_benchmark /caminho/foto-rgb16.tif
```

Para incluir CUDA na comparação no mesmo processo:

```sh
cmake -S tools/opencl-tests -B /tmp/art-opencl-compare -DCMAKE_BUILD_TYPE=Release \
  -DART_COMPARE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/art-opencl-compare -j
OMP_NUM_THREADS=28 ART_CUDA=1 /tmp/art-opencl-compare/gpu_benchmark /caminho/foto-rgb16.tif
```

O benchmark exige TIFF RGB16 opaco, intercalado, com dimensões de pelo menos 1024 × 1024 e no máximo 64 Mi pixels. Requer execução efetiva dos backends selecionados: erro ou indisponibilidade não vira um tempo de fallback CPU apresentado como GPU. Cada linha de stdout é um objeto JSON; informações do dispositivo e erros vão para stderr.

Nesta máquina, os argumentos adicionais de descoberta foram `-DOpenCL_INCLUDE_DIR=/tmp/art-opencl-deps/root/usr/include -DOpenCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so.1`. Para confirmar o loader do sistema, a rodada final usou `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`; isso foi específico do ambiente, que também disponibiliza um loader no toolkit. `86` é a arquitetura CUDA da A4500, não uma exigência do build OpenCL.

Logs locais: `/tmp/art-opencl-only-ctest.log`, `/tmp/art-opencl-cuda-regression.log` e `/tmp/art-opencl-comparison-system-icd.{jsonl,stderr}`. O JSON versionável em `doc/benchmarks` preserva os resultados, pois `/tmp` é temporário.
