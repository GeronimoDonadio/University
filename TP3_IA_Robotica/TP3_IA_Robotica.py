import sqlite3
import random
import pyopencl as cl
import numpy as np

# Configurar el contexto y la cola de comandos de OpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Crear la base de datos
def crear_base_datos():
    conn = sqlite3.connect('agente_avanzado.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS decisiones (
        id INTEGER PRIMARY KEY,
        accion TEXT,
        contexto TEXT,
        prioridad INTEGER,
        probabilidad REAL,
        consecuencia TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historial (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        accion TEXT,
        contexto TEXT,
        resultado TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    decisiones = [
        ('acción_1', 'contexto_1', 1, 0.5, 'consecuencia_positiva'),
        ('acción_2', 'contexto_2', 2, 0.3, 'consecuencia_negativa'),
        ('acción_3', 'contexto_1', 3, 0.2, 'consecuencia_neutra'),
        ('acción_4', 'contexto_3', 1, 0.4, 'consecuencia_positiva'),
        ('acción_5', 'contexto_2', 2, 0.2, 'consecuencia_positiva'),
        ('acción_6', 'contexto_3', 1, 0.3, 'consecuencia_negativa')
    ]
    cursor.executemany('''
    INSERT INTO decisiones (accion, contexto, prioridad, probabilidad, consecuencia)
    VALUES (?, ?, ?, ?, ?)
    ''', decisiones)
    conn.commit()
    conn.close()

# Kernel de OpenCL para ajustar probabilidades
kernel_code = """
__kernel void ajustar_probabilidades(
    __global float* probabilidades,
    __global int* resultados,
    __global float* frecuencias,
    int num_decisiones
) {
    int gid = get_global_id(0);
    if (gid < num_decisiones) {
        if (resultados[gid] == 1) {
            probabilidades[gid] = min(1.0f, probabilidades[gid] + 0.2f * frecuencias[gid]);
        } else if (resultados[gid] == -1) {
            probabilidades[gid] = max(0.05f, probabilidades[gid] - 0.2f * frecuencias[gid]);
        }
    }
}
"""

# Compilar el kernel
program = cl.Program(ctx, kernel_code).build()

# Función para registrar el historial de decisiones
def registrar_historial(accion, contexto, resultado):
    conn = sqlite3.connect('agente_avanzado.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO historial (accion, contexto, resultado)
    VALUES (?, ?, ?)
    ''', (accion, contexto, resultado))
    conn.commit()
    conn.close()

# Función para obtener una acción aleatoria basada en contexto
def obtener_accion_aleatoria(contexto_actual):
    conn = sqlite3.connect('agente_avanzado.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT accion, prioridad, probabilidad, consecuencia
    FROM decisiones
    WHERE contexto = ?
    ORDER BY prioridad DESC
    ''', (contexto_actual,))
    decisiones = cursor.fetchall()
    conn.close()

    if not decisiones:
        return "No hay decisiones disponibles para este contexto."

    acciones, prioridades, probabilidades, consecuencias = zip(*decisiones)
    prioridades = list(map(int, prioridades))  # Convertir prioridades a int
    probabilidades = [float(p) if isinstance(p, (float, int)) else float.fromhex(p.hex()) for p in probabilidades]

    ponderaciones = [max(0.05, p * pr) for p, pr in zip(probabilidades, prioridades)]
    accion_seleccionada = random.choices(acciones, weights=ponderaciones, k=1)[0]
    consecuencia = consecuencias[acciones.index(accion_seleccionada)]
    resultado = random.choice(['exito', 'fracaso', 'neutro'])
    registrar_historial(accion_seleccionada, contexto_actual, resultado)
    return accion_seleccionada, consecuencia, resultado

# Ejecutar múltiples iteraciones con OpenCL para ajustar probabilidades
def ejecutar_multiples_iteraciones(contextos, iteraciones_por_contexto=100_000):
    conn = sqlite3.connect('agente_avanzado.db')
    cursor = conn.cursor()
    for contexto in contextos:
        print(f"\nContexto actual: {contexto}")
        for i in range(iteraciones_por_contexto):
            accion, consecuencia, resultado = obtener_accion_aleatoria(contexto)
            if (i + 1) % 10_000 == 0:
                print(f"Iteración {i+1}: {accion} con consecuencia: {consecuencia}, resultado: {resultado}")

        cursor.execute("SELECT probabilidad FROM decisiones WHERE contexto=?", (contexto,))
        # Conversión explícita de bytes a float
        probabilidades = [float(p[0]) if isinstance(p[0], (float, int)) else float.fromhex(p[0].hex()) for p in cursor.fetchall()]
        probabilidades = np.array(probabilidades, dtype=np.float32).flatten()

        resultados = np.array([1 if r == 'exito' else -1 if r == 'fracaso' else 0 for r in cursor.execute("SELECT resultado FROM historial WHERE contexto=?", (contexto,))], dtype=np.int32)
        frecuencias = np.ones_like(probabilidades, dtype=np.float32)

        # Crear buffers en la GPU
        probabilidades_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=probabilidades)
        resultados_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=resultados)
        frecuencias_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=frecuencias)

        # Ejecutar el kernel
        program.ajustar_probabilidades(queue, probabilidades.shape, None, probabilidades_buf, resultados_buf, frecuencias_buf, np.int32(len(probabilidades)))
        cl.enqueue_copy(queue, probabilidades, probabilidades_buf)

        # Actualizar probabilidades en la base de datos
        for j, probabilidad in enumerate(probabilidades):
            cursor.execute("UPDATE decisiones SET probabilidad=? WHERE id=?", (probabilidad, j + 1))
        conn.commit()

    conn.close()

# Inicializar la base de datos y ejecutar iteraciones
crear_base_datos()
contextos = ['contexto_1', 'contexto_2', 'contexto_3']
ejecutar_multiples_iteraciones(contextos, iteraciones_por_contexto=100_000)
