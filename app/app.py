from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Lista de usuários (apenas para fins de demonstração)
users = [
    {'username': 'usuario1', 'password': 'senha1'},
    {'username': 'usuario2', 'password': 'senha2'},
]

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        for user in users:
            if user['username'] == username and user['password'] == password:
                # Autenticação bem-sucedida, redirecione para outra página após o login
                return redirect(url_for('dashboard'))

        # Se a autenticação falhar, exiba uma mensagem de erro
        error_message = 'Credenciais inválidas. Tente novamente.'
        return render_template('login.html', error_message=error_message)

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    # Lógica da outra página após o login
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=6060)
