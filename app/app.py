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

def autenticar_usuario(username, password):
    for usuario in usuarios:
        if usuario['name'] == username and usuario['password'] == hash_senha(password):
            return True
    return False

# Resto do seu código permanece o mesmo



@app.route('/')
def index():
    return redirect(url_for('login'))

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

@app.route('/dashboard')
def home():
    # Lógica da outra página após o login
    return render_template('dashboard.html')


@app.route('/cadastrar', methods=['GET', 'POST'])
def cadastrar():
    if request.method == 'POST':
        # Obtenha os dados do formulário
        username = request.form.get('name')
        password = request.form.get('password')

        # Verifique se o usuário já existe
        if any(usuario['name'] == username for usuario in usuarios):
            error_message = "Este nome de usuário já está em uso."
            return render_template('cadastrar.html', error_message=error_message)

        # Adicione o novo usuário ao arquivo JSON
        new_user = {'name': username, 'password': hash_senha(password)}
        usuarios.append(new_user)

        with open('bd.json', 'w') as json_file:
            json.dump(usuarios, json_file, indent=4)

        # Redirecione para a página de login após o cadastro bem-sucedido
        return redirect(url_for('login'))

    # Se o método for GET, simplesmente renderize a página de cadastro
    return render_template('cadastrar.html')
if __name__ == '__main__':
    app.run(debug=True, port=6060)
