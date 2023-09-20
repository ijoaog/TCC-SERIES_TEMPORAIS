import hashlib
import json

def hash_senha(password):
    hash_obj = hashlib.sha256()
    hash_obj.update(password.encode('utf-8'))
    senha_hash = hash_obj.hexdigest()
    return senha_hash

# Carregar os dados do arquivo JSON
with open('./../../app/bd.json', 'r') as json_file:
    dados_json = json.load(json_file)

nomes = ["banana", "bola", "Pedro"]

for nome in nomes:
    for item in dados_json:
        if item['password'] == hash_senha(nome):
            print(f"Nome de usuário correspondente: {item['name']}")
            print(f"Senha hash correspondente: {item['password']}")
