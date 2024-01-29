# ai-chatbot
- Build AI Conversational Chatbot leveraging LLM

## Initial BE Setup
- This repo currently contains the starter files.

- Clone repo and create a virtual environment
    ```
    python3 -m venv .venv
    . .venv/bin/activate # for Mac
    .venv\Scripts\activate # for Window
    ```
    Install dependencies
    ```
    $ (venv) pip install -r requirements.txt
    ```
- Install nltk package
    ```
    $ (venv) python
    >>> import nltk
    >>> nltk.download('punkt')
    ```
## Playground BE

- Modify `intents.json` with different intents and responses for your Chatbot

- Run train script: This will dump data.pth file. And then run
the following command to test it in the console.
    ```
    $ (venv) python train.py
    ```
- Run python demo examples
    ```
    $ (venv) python chat.py
    ```
## Launch BE
```
cd ./backend
python app.py
```

## Launch FE
```
cd ./fontend
Click 'Go Live'
```

## Setup PostgreSQL using Docker 
- Install pgvector and postgreSQL using pgvector docker image
```
docker pull ankane/pgvector
docker run --name pgvector-demo -e POSTGRES_PASSWORD=123456 -p 5432:5432 -d ankane/pgvector
```
- Start an existing container
```
docker start pgvector-demo
```
- Attach termina to the running container:
```
docker attach pgvector-demo
```
- Install a GUI tool such pgAdmin or use psql

## Setup pgAdmin
- Click on 'Servers', choose 'Create', choose 'Server Group', give it a name, for example 'Pigment'. Then server group 'Pigment' is created.
- Click on 'Pigment' server, choose 'Resiter', choose 'Server...', 
    - In 'General' tab, give it a name, e.g. 'pgvector_test'
    - In 'Connection' tab:
        - Host name/adress: localhost
        - Password: 123456
    - Click 'Save', then 'pgvector_test' register is created.
- Click on 'pgvector_test', then right click on 'Database', then 'Create'
    - Give 'General' tab a name, e.g. 'vector_db', then 'Save'
- Enable 'vector' extention:
    - Right click on 'vector_db', choose 'Query Tool'
    - Add:
        ```
        CREATE EXTENSION vector
        ```
