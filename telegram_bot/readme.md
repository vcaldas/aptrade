https://docs.python-telegram-bot.org/en/stable/


# Bot de Telegram com FastAPI

Este é um projeto que implementa um bot de Telegram utilizando o framework web FastAPI para fornecer informações sobre notícias. O bot é capaz de responder a comandos específicos e fazer solicitações a uma API FastAPI para obter as notícias.

## Pré-requisitos

Certifique-se de ter instalado o Python em sua máquina. Além disso, as seguintes bibliotecas devem estar instaladas:

- `python-telegram-bot`: Uma biblioteca que fornece uma interface para a API do Telegram.
- `httpx`: Uma biblioteca para fazer solicitações HTTP.

Você pode instalar as dependências usando o arquivo `requirements.txt` fornecido no projeto.

## Configuração

Antes de executar o bot, é necessário configurar o token do bot do Telegram. Siga os seguintes passos:

1. Crie um novo bot no Telegram conversando com o @BotFather.
2. Copie o token fornecido pelo @BotFather.
3. Defina o token do bot como uma variável de ambiente com o nome "TOKEN" em seu sistema operacional.

3. Execute o script `init.sh` para iniciar o projeto:

```

Este script irá:

- Criar o arquivo `.env` com as variáveis de ambiente necessárias.
- Instalar as dependências do Python listadas em `requirements.txt`.
- Fazer o build das imagens Docker usando o `docker-compose`.
- Iniciar os contêineres com `docker-compose up`.
- Executar os testes automatizados com pytest dentro do contêiner `web`.
- Desligar e remover os contêineres após a execução.

Certifique-se de que o Docker e o docker-compose estejam instalados e configurados corretamente no seu sistema antes de executar o script.
```
4. Agora, o projeto estará em execução, e você poderá acessar a API FastAPI e interagir com o bot Telegram para obter notícias do RSS.


## Executando o Bot

Para iniciar o bot, execute o seguinte comando:

```bash
python bot.py
```

```bash
docker-compose build
docker-compose up
```

## O bot estará pronto para receber comandos no Telegram.



## Comandos Disponíveis
```
/start: Inicia a interação com o bot e recebe uma mensagem de boas-vindas.
/help: Exibe uma mensagem de ajuda.
/noticia: Obtém a primeira notícia do feed da API FastAPI e envia o título para o usuário.
/noticia_t: Obtém as três primeiras notícias do feed da API FastAPI e envia os títulos para o usuário.
/noticias: Obtém todas as notícias do feed da API FastAPI e envia os títulos e links para o usuário.
Funcionamento
O bot, ao receber um comando /noticia, /noticia_t ou /noticias, faz uma solicitação HTTP à API FastAPI que está sendo executada em outro serviço. A API FastAPI obtém as notícias de um feed RSS do Google News e retorna os dados em formato JSON. O bot processa os dados recebidos e envia as informações de volta ao usuário do Telegram.
```
Licença
Este projeto está sob a licença MIT, o que significa que você pode modificá-lo e usá-lo livremente para seus próprios fins.

Aviso
Este projeto é apenas um exemplo didático e não foi projetado para uso em produção. Tenha cuidado ao utilizar o código em ambientes de produção e considere as melhores práticas de segurança ao lidar com solicitações externas.


Lembre-se de substituir as informações necessárias no `README.md`, como a descrição do projeto, os comandos disponíveis, as instruções de configuração e execução, bem como quaisquer outros detalhes relevantes para o seu projeto específico. O arquivo `README.md` é uma ferramenta importante para compartilhar informações sobre o projeto e ajudar os colaboradores a entenderem como funciona e como contribuir para ele.
