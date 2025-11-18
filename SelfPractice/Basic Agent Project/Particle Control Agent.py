import subprocess
import sys

env_process = subprocess.Popen(['python', 'Controllable Particle.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

while True:
    state_string = env_process.stdout.readline().decode('utf-8').strip()
    print(state_string)
    sys.stdout.flush()

    command_to_send = input('Command: ')
    if command_to_send == 'quit':
        break

    env_process.stdin.write(f'{command_to_send}\n'.encode('utf-8'))
    env_process.stdin.flush()  
    
env_process.terminate()