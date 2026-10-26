# GPU: exibição, cache da foto e redimensionamento

Experimentos de 26/10/2026 na RTX A4500, driver 615.71.09, CUDA 13.4 e dois Xeon E5-2680 v4. CPU com 28 threads. Dados completos: [medições em JSON](benchmarks/gpu-display-resize-20261026.json).

## Resultado e decisão

**Cache na VRAM faz sentido quando a mesma imagem é reutilizada. Copiar uma imagem para executar apenas uma operação pode custar mais que processá-la na CPU.** Os testes abaixo incluem as transferências pertinentes, alocações, validação e entrega do resultado.

- **Exibição:** uma textura residente da foto inteira acelerou o desenho de uma região visível. Entretanto, substituir a exibição da prévia pequena por OpenGL misturado com Cairo piorou o tempo nesta implementação.
- **Redimensionamento:** Lanczos CUDA com upload em cada chamada perdeu para a CPU. Com a origem residente, houve ganho de aproximadamente **5,9×** para 1683 × 1123. Para 2560 × 1708 não houve ganho consistente, mesmo reutilizando a origem.
- **Integração:** a apresentação OpenGL permanece em um programa de teste independente. A API CUDA de cache/redimensionamento foi implementada, mas **não é chamada pelo pipeline do editor nem pela exportação**. Não há uma nova opção que ative essas duas rotas no ART. `ART_CUDA=1` continua ativando somente o bloco de deconvolução já documentado.

Assim, não se adicionou um caminho mais lento à utilização habitual. O próximo passo justificável para apresentação seria um desenho que reutilize a textura ao navegar pela mesma imagem e limite o custo das sobreposições, seguido de medição dentro do editor completo.

## Dados e método

Origem: TIFF RGB16 de **6732 × 4492**, previamente exportado de `276A0950.cr2` da pasta fornecida pelo usuário. Os RAWs e perfis originais não foram alterados. Os arquivos de trabalho e capturas ficaram em `/tmp`.

Para Lanczos, o TIFF foi convertido em três planos FP32. Mede-se o redimensionador, sem decodificação RAW, conversão RGB/Lab, ICC ou gravação de arquivo. A ordem vertical → horizontal, os coeficientes calculados na CPU e a aritmética FP32 seguem o ART. O teste usa a mesma função CPU chamada por `ImProcFunctions::Lanczos`; esta função foi extraída para `rtengine/lanczos.cc`. A normalização horizontal passou a percorrer somente os coeficientes válidos, evitando ler posições não inicializadas que não participavam da saída.

Para exibição, o mesmo TIFF foi convertido para RGB8, como o formato usado pela prévia atual. O teste criou uma **janela GTK3 real no X11**, com renderer confirmado como `NVIDIA RTX A4500/PCIe/SSE2`. Comparou Cairo e OpenGL com o mesmo recorte, translação e uma sobreposição Cairo. Não foi um teste em Xvfb nem em renderizador de software.

As medições finais foram executadas sequencialmente, sem compilação ou exportação concorrente. Não representam FPS do editor, latência de interação até o monitor ou velocidade de exportação completa.

## Exibição e textura residente

Janela de 1150 × 750; região da foto de até 1100 × 700. Cada modo tem cinco quadros de aquecimento e trinta amostras. Tempo do início de `draw` até `after-paint`, incluindo sincronização do contexto GL corrente e do display GDK. A captura para comparar pixels ocorre fora do intervalo medido.

| Fonte disponível ao desenho | Cairo antes de inicializar GL | GL com upload a cada quadro | GL com textura residente | Cairo após inicializar GL |
| --- | ---: | ---: | ---: | ---: |
| Prévia de 1100 × 733 | 1,82 ms | 9,84 ms | 8,50 ms | 5,99 ms |
| Foto inteira de 6732 × 4492, mostrando um recorte | 48,09 ms | 96,44 ms | 8,55 ms | 54,87 ms |

São medianas. A coluna final mostra que inicializar GL também muda o custo posterior da composição GDK/Cairo; não se escolheu apenas essa referência mais lenta para anunciar ganho. O primeiro quadro GL, incluindo criação de contexto e primeira transferência, levou **203 ms** na prévia e **464 ms** na foto inteira. Esses custos não estão escondidos nas medianas após aquecimento.

O ganho de aproximadamente **5,6×** com a foto inteira é específico desse ensaio: Cairo recebe o pixbuf inteiro e paga sua preparação, embora só uma região seja desenhada. **O ART atual já produz recortes para a tela**, portanto esse número não pode ser aplicado diretamente ao editor. No teste da prévia pequena, a mistura de GL com sobreposições Cairo perdeu claramente.

A textura da foto ocupa até aproximadamente **115,4 MiB**, contabilizando quatro bytes por pixel mesmo sendo RGB8; o driver pode armazená-la como RGBA8. A prévia ocupa até 3,08 MiB. O protótipo limita cada cache a 256 MiB e ao tamanho máximo de textura do dispositivo.

`GpuPreview` mantém uma textura por instância. Dimensões, identidade da origem e uma **geração explícita** controlam o upload; somente o endereço do buffer não é uma chave válida. Mudar pixels, foto, processamento ou transformação de monitor exige uma nova geração. `clear()` libera a textura/contexto. A classe pertence à thread GTK; não é um cache global de todas as fotos abertas.

Foram comparados tanto os bytes da textura lidos de volta quanto os pixels capturados da janela: **zero diferenças**, incluindo orientação vertical, recorte e sobreposição. Também se alterou o conteúdo mantendo o mesmo buffer e verificou-se a atualização por geração. Isso valida o transporte RGB8; não substitui testes de gerenciamento de cor e interação no editor. Uma integração futura deve receber a imagem **depois** de `rgb2monitor`, preservando ICC e soft proof.

A API [GDK `cairo_draw_from_gl`](https://docs.gtk.org/gdk3/func.cairo_draw_from_gl.html) permite compor uma textura com Cairo, mas pode recorrer a readback em determinadas condições. Ao desenhar Cairo sobre uma região GL é necessário sinalizá-la com [`mark_paint_from_clip`](https://docs.gtk.org/gdk3/method.Window.mark_paint_from_clip.html). O protótipo sinaliza conservadoramente o clip inteiro: a sobreposição fica correta, mas tem custo. O suporte atual é apenas OpenGL desktop com fator de escala 1; ausência de GL e HiDPI retornam ao desenho Cairo, ambos testados.

## Lanczos CUDA com cache da imagem de origem

Medianas de quatro repetições após a primeira de cada tamanho; as cinco repetições e a inicialização estão preservadas no JSON. A coluna residente **inclui download e publicação em RAM**, não apenas o tempo do kernel.

| Saída a partir de 30,24 MP | CPU | CUDA com nova cópia da origem | CUDA com origem já residente | VRAM origem + trabalho |
| --- | ---: | ---: | ---: | ---: |
| 2560 × 1708 | 24,68 ms | 101,46 ms | 25,94 ms | 528,0 MiB |
| 1683 × 1123 | 36,12 ms | 88,65 ms | 6,17 ms | 454,5 MiB |

O upload/preparação da origem levou aproximadamente **79–82 ms** com o contexto pronto. A primeira chamada do processo, incluindo inicialização CUDA, levou **338 ms** para 2560 × 1708. Em uma rodada anterior, CPU/GPU residentes nessa saída ficaram perto de 29/27 ms: essa variação reforça que não há ganho confiável nesse caso.

Para 1683 × 1123, a amortização estimada com contexto já pronto ocorre em cerca de **três redimensionamentos da mesma origem**: compara-se `88,65 + (N−1) × 6,17` com `N × 36,12`. É uma estimativa derivada das medições, não uma política automática nem garantia para diferentes escalas, hardware ou fotos. Alterar o processamento que gerou a origem exige outro upload e reinicia essa conta.

`rtengine::cuda::ResizeImage` oferece `upload`, `resize`, `deviceBytes` e `clear`. Cada instância pertence a uma thread e guarda somente uma imagem planar FP32 decodificada. `upload` invalida a origem anterior mesmo se a nova carga falhar; reutilizar o endereço de RAM nunca reaproveita pixels antigos implicitamente. O intermediário e a saída são reutilizados, com limite total de **1 GiB por instância**, até 64 Mi pixels por imagem e escalas entre 0,05 e 4. A memória host de preparação continua necessária.

Resultado inválido, recurso indisponível ou falha de execução não publica alterações na saída. O teste verifica substituição da origem, mudança de dimensões da saída, liberação, entradas não finitas e limites. Os kernels usam FP32 sem contração FMA; **não utilizam Tensor Cores**.

## O que colocar no cache

Para exibição, vale guardar **a imagem já convertida para o monitor** como textura RGB8. Para reutilizar operações, vale guardar **pixels decodificados/processados** no formato que essas operações consomem, como os planos FP32 deste experimento. A origem FP32 de 30,24 MP, sozinha, custa aproximadamente 346 MiB.

Copiar os bytes comprimidos do arquivo `.cr2` para a VRAM não acelera o decodificador atual, que executa na CPU. Pode até exigir outra transferência para trazê-los de volta. O cache precisa corresponder a uma etapa reutilizável do processamento, com invalidação por foto, parâmetros e perfil de cor, e liberação ao encerrar seu uso. Ele tampouco elimina automaticamente a cópia em RAM necessária às operações CPU e às ferramentas do editor.

## Validação e reprodução

- CTest: **4/4** testes de deconvolução e resize, incluindo GPU indisponível.
- Resize: campos constantes, ruído, valores negativos, bordas fortes, linhas com padding, redução e ampliação. Também foi comparada a imagem real de 30,24 MP. **Zero diferenças FP32** entre CPU e GPU nos casos executados.
- Compute Sanitizer: teste de resize com `memcheck` e verificação de vazamentos: **zero erros e zero bytes vazados**.
- Refatoração CPU: exportação RGB16 de 2560 × 1708 com uma thread, comparada com o executável anterior à extração do Lanczos: **zero amostras diferentes**, tanto no build com CUDA desativado em runtime quanto no build compilado sem CUDA.
- Apresentação: textura, janela, sobreposição e invalidação verificadas; fallback testado com `GDK_GL=disable` e `GDK_SCALE=2`. A interface completa do ART não recebeu esse protótipo.

Teste CUDA, com toolkit e dependências de desenvolvimento instalados:

```sh
cmake -S tools/cuda-tests -B /tmp/art-cuda-tests -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build /tmp/art-cuda-tests -j
ctest --test-dir /tmp/art-cuda-tests --output-on-failure
OMP_NUM_THREADS=28 /tmp/art-cuda-tests/resize_test /caminho/foto-rgb16.tif
OMP_NUM_THREADS=28 /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --leak-check full --error-exitcode 99 /tmp/art-cuda-tests/resize_test
```

Teste de exibição independente de CUDA; requer GTK3, epoxy, um display acessível e OpenGL desktop. Abre uma janela temporária e encerra automaticamente:

```sh
cmake -S tools/gpu-display-tests -B /tmp/art-display-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/art-display-tests -j
/tmp/art-display-tests/display_test /caminho/foto.tif
/tmp/art-display-tests/display_test /caminho/foto.tif full
ART_TEST_EXPECT_NO_GL=1 GDK_GL=disable /tmp/art-display-tests/display_test /caminho/foto.tif
ART_TEST_EXPECT_NO_GL=1 GDK_SCALE=2 /tmp/art-display-tests/display_test /caminho/foto.tif
```

`ART_TEST_CAPTURE=1` salva capturas de diagnóstico em `/tmp/art-display-phase-*.png`; evite execuções simultâneas com essa opção. Use imagem opaca, pelo menos 64 × 64. Confirme o renderer no log: passar em software não demonstra aceleração por GPU.

Logs desta sessão: `/tmp/art-resize-final-real.log`, `/tmp/art-gpu-expanded-ctest.log`, `/tmp/art-resize-memcheck.log`, `/tmp/art-display-final-{preview,full,fallback,hidpi}.log`. Os resultados essenciais foram preservados no JSON do repositório, pois `/tmp` é temporário.
