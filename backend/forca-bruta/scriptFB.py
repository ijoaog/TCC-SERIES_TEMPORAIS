import hashlib
import json
import time

def hash_senha(password):
    hash_obj = hashlib.sha256()
    hash_obj.update(password.encode('utf-8'))
    senha_hash = hash_obj.hexdigest()
    return senha_hash

# Carregar os dados do arquivo JSON
with open('./../../app/bd.json', 'r') as json_file:
    dados_json = json.load(json_file)

with open('./senhas3.txt', 'r') as arquivo:
    nomes = arquivo.read().splitlines()

start_time = time.time()  # Grava o tempo inicial

correspondencias_encontradas = set()  # Conjunto para armazenar correspondências únicas

for nome in nomes:
    correspondencia_encontrada = False  # Variável de controle
    for item in dados_json:
        if item['password'] == hash_senha(nome) and item['name'] not in correspondencias_encontradas:
            end_time = time.time()  # Grava o tempo final quando a correspondência é encontrada
            tempo_decorrido = end_time - start_time  # Calcula o tempo decorrido
            print(f"Nome de usuário correspondente: {item['name']}")
            print(f"Senha hash correspondente: {item['password']}")
            print(f"Senha hash correspondente: {item['senha_sem_hash']}")
            print(f"Tempo decorrido: {tempo_decorrido} segundos")
            correspondencias_encontradas.add(item['name'])  # Adicione o nome à lista de correspondências encontradas
            correspondencia_encontrada = True  # Define a variável como verdadeira
            # Você pode adicionar mais ações aqui se quiser
            break  # Sai do loop interno quando a correspondência é encontrada

    if correspondencia_encontrada:
        continue  # Continua para o próximo nome no arquivo senhas.txt

    # # Se a correspondência não for encontrada para um nome, você pode mostrar uma mensagem
    # print(f"Nenhuma correspondência encontrada para o nome: {nome}")

# Indica que todas as verificações foram concluídas
print("Todas as verificações foram concluídas.")