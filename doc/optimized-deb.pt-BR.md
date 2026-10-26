# Pacote Debian do ART otimizado

Pacote local `art-optimized`, versão `2026.10.26+git4a7ed4fb7-1`, para Debian 13 (trixie), amd64. Inclui as alterações não commitadas desta árvore, GUI, CLI, recursos, perfis e preset otimizado. O hash Git identifica a base, não representa sozinho o conteúdo modificado.

## Instalação e uso

Na raiz do projeto:

```sh
sudo apt install ./dist/art-optimized_2026.10.26+git4a7ed4fb7-1_amd64.deb
```

O APT resolve as bibliotecas do Debian, incluindo Lensfun e sua base de lentes. Não é necessário instalar o toolkit CUDA: seu runtime está vinculado estaticamente. A aceleração exige um driver NVIDIA compatível com CUDA 13.4; caso indisponível, as operações retornam à CPU.

Abra **ART Otimizado** no menu de aplicativos ou execute:

```sh
art-optimized -N
art-optimized-cli -h
```

O atalho usa `-N` para abrir esta versão em uma nova instância, sem encaminhar a solicitação a outro ART já aberto. Para usar somente CPU:

```sh
ART_CUDA=0 art-optimized -N
```

O preset é aplicado automaticamente pelos dois comandos; variáveis de ambiente explícitas continuam prevalecendo. O monitor permanece na barra superior existente. Configuração e cache mantêm o sufixo `_gpu_testing`, usado nos testes anteriores; a instalação não copia nem apaga preferências ou fotos.

Os binários e recursos ficam em `/usr/lib/art-optimized`, com lançadores em `/usr/bin` e documentação em `/usr/share/doc/art-optimized`. O nome do pacote, comandos, ícones e atalho são próprios, permitindo coexistência com um ART instalado por outro pacote. Para remover: `sudo apt remove art-optimized`.

## Compatibilidade

Esta compilação usa `-march=native` no computador de Paulo (Xeon E5-2680 v4, Broadwell) e CUDA SM 8.6, correspondente à RTX A4500. Não é um binário universal para CPUs amd64 antigas ou uma versão validada em outras distribuições/GPU. As otimizações e correções são as descritas em `editor-performance.pt-BR.md` e `advanced-performance.pt-BR.md`; o empacotamento não introduz outro algoritmo de processamento.

## Reprodução

O pacote foi extraído e testado em pasta isolada, sem instalação no sistema: GUI com 12 trocas de pasta e fechamento normal, exportação RAW com CUDA e exportação com GPU ocultada para confirmar fallback CPU. Os dados RAW decodificados permaneceram iguais. O APT simulou a instalação sem remover pacotes existentes; os 673 arquivos foram conferidos por checksum. A ligação estática altera a distribuição do runtime, sem mudar os kernels usados. Os resultados detalhados acompanham o pacote em `dist/art-optimized-validation.json`.

Dependências de desenvolvimento precisam estar instaladas ou acessíveis pelo `PKG_CONFIG_PATH`. Nesta sessão, parte delas está extraída em `/tmp/art-build-deps`, utilizada somente para compilar e validar; esses caminhos não são necessários após a instalação do pacote.

```sh
PKG_CONFIG_PATH=/tmp/art-build-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
cmake -S . -B /tmp/art-deb-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_BUNDLE=ON \
  -DBUNDLE_BASE_INSTALL_DIR=/usr/lib/art-optimized \
  -DDATADIR=. -DDOCDIR=doc -DCREDITSDIR=. -DLICENCEDIR=. \
  -DDESKTOPDIR=share/applications -DICONSDIR=share/icons \
  -DCACHE_NAME_SUFFIX=_gpu_testing -DPROC_TARGET_NUMBER=2 \
  -DENABLE_CTL=OFF -DENABLE_OCIO=OFF -DENABLE_MIMALLOC=OFF \
  -DENABLE_LIBRAW=ON -DART_ENABLE_CUDA=ON -DART_CUDA_STATIC_RUNTIME=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_INSTALL_RPATH= \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
cmake --build /tmp/art-deb-build --target art art-cli -j20
python3 tools/build_optimized_deb.py \
  --build-dir /tmp/art-deb-build --output dist \
  --version 2026.10.26+git4a7ed4fb7-1 \
  --deps-root /tmp/art-build-deps --cuda-license /usr/local/cuda/EULA.txt
```

O gerador usa `DESTDIR`, calcula dependências com `dpkg-shlibdeps`, verifica o atalho e rejeita binários com RPATH/RUNPATH ou dependência de `libcudart.so`. Não instala nada no sistema. O `.deb` é acompanhado por SHA-256 e um manifesto de compilação. Se todas as dependências de desenvolvimento estiverem instaladas no sistema, omita `--deps-root`.
