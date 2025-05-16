import os
import subprocess
"""def iswin(a):
    if "Loss" in a[8]:
        return False
    else:
        return True"""

def eliminar_perdidas():
    """
    Elimina las filas que contienen pérdidas.
    """
    lista_de_partidas = os.listdir("pacman_data")
    lista_de_partidas.sort()
    for i in lista_de_partidas:
        output =  subprocess.run(["python", "pacman.py", "--csv" ,f"./pacman_data/{i}", "-q"], capture_output = True).stdout
        info_partida = output.split()
        #print(info_partida)
        info_partida = info_partida[-1].decode('utf-8')
        if info_partida == 'Loss':
            print(f"Eliminando {i}")
            os.remove(f"pacman_data/{i}")
        # else:
        #     print(f"Conservando {i}")
    

eliminar_perdidas()