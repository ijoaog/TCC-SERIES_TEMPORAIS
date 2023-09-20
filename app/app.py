from flask import Flask, jsonify, make_response, render_template, request, redirect, url_for
import hashlib
import json

app = Flask(__name__)

def hash_senha(password):
    hash_obj = hashlib.sha256()
    hash_obj.update(password.encode('utf-8'))
    senha_hash = hash_obj.hexdigest()
    return senha_hash

# Carregue os usuários do arquivo JSON
with open('bd.json', 'r') as json_file:
    usuarios = json.load(json_file)

# Classe de usuário para gerenciar nome de usuário e senha
class Usuario:
    def __init__(self, username, password):
        self.username = username
        self.password = password

# Lista de usuários com senhas hashadas (opcional)
# Usaremos os usuários do JSON em vez disso.
# usuarios = [
#     Usuario('usuario1', hash_senha('senha1')),
#     Usuario('usuario2', hash_senha('senha2')),
#     Usuario('usuario3', hash_senha('senha3')),
# ]
#hash_senha(password)
def autenticar_usuario(username, password):
    for usuario in usuarios:
        if usuario['name'] == username and usuario['password'] == hash_senha(password):
            return True
    return False

# Resto do seu código permanece o mesmo



@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if autenticar_usuario(username, password):
            # Autenticação bem-sucedida, redirecione para outra página após o login
            return redirect(url_for('home'))
        else:
            error_message = "Usuário ou senha incorretos."
            return render_template('login.html', error_message=error_message)

    return render_template('login.html')

@app.route('/home')
def home():
    # Lógica da outra página após o login
    return render_template('home.html')

@app.route('/usuarios', methods=['GET'])
def get_usuarios():
    return make_response(
        jsonify(usuarios)
    )

@app.route('/cadastrar', methods=['GET', 'POST'])
def cadastrar():
    if request.method == 'POST':
        # Obter os dados do formulário
        name = request.form.get('name')
        password = request.form.get('password')

        # Verificar se o nome de usuário já existe no JSON
        for usuario in usuarios:
            if usuario['name'] == name:
                error_message = "Nome de usuário já existe."
                return render_template('cadastrar.html', error_message=error_message)

        # Criar um novo usuário e adicionar ao JSON
        novo_usuario = {
            'name': name,
            'password': hash_senha(password)
        }
        usuarios.append(novo_usuario)

        # Atualizar o arquivo JSON com o novo usuário
        with open('bd.json', 'w') as json_file:
            json.dump(usuarios, json_file, indent=4)

        return redirect(url_for('login'))

    return render_template('cadastrar.html')



@app.route('/logout', methods=['POST'])
def logout():
    # Lógica de logout aqui (por exemplo, limpar a sessão)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=6060)
