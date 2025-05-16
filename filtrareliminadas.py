import os
def iswin(a):
    if "Loss" in a[8]:
        return False
    else:
        return True

def eliminar_perdidas():
    """
    Elimina las filas que contienen pérdidas.
    """
    lista_de_partidas = os.listdir("pacman_data")
    lista_de_partidas.sort()
    for i in lista_de_partidas:
        info_partida = !python pacman.py --csv pacman_data/{i} -q
        if iswin(info_partida) == False:
            print(f"Eliminando {i}")
            os.remove(f"pacman_data/{i}")
        # else:
        #     print(f"Conservando {i}")
    

eliminar_perdidas()