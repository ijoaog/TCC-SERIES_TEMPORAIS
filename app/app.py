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

@app.route('/home')
def home():
    # Lógica da outra página após o login
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=6060)
