import os
import pandas as pd


def combinar_metricas_en_excel(base_dir, output_file):
    """
    Combina métricas de los clientes y el servidor en un archivo Excel, calculando promedios.

    Args:
        base_dir (str): Directorio base donde están las carpetas de resultados.
        output_file (str): Ruta donde se guardará el archivo Excel combinado.
    """
    # Identificar subdirectorios de clientes y servidor
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    all_metrics = []

    # Procesar cada subdirectorio
    for subdir in subdirs:
        source_name = os.path.basename(subdir)  # Nombre del cliente o servidor
        metrics_files = [
            os.path.join(subdir, f)
            for f in os.listdir(subdir)
            if f.endswith(".csv")
        ]

        for metrics_file in metrics_files:
            cliente_metrics = pd.read_csv(metrics_file)

            # Extraer número de ronda desde el nombre del archivo
            try:
                ronda_number = int(metrics_file.split("_")[-1].replace(".csv", ""))
            except ValueError:
                print(f"Error procesando el archivo {metrics_file}. Asegúrate de que incluye 'ronda_<número>.csv'")
                continue

            cliente_metrics["Ronda"] = ronda_number
            cliente_metrics["Origen"] = source_name
            all_metrics.append(cliente_metrics)

    # Combinar todas las métricas
    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # Calcular promedio por origen (Servidor y cada cliente)
    avg_metrics = combined_metrics.groupby("Origen").mean()

    # Guardar en Excel
    with pd.ExcelWriter(output_file) as writer:
        combined_metrics.to_excel(writer, sheet_name="Métricas Completas", index=False)
        avg_metrics.to_excel(writer, sheet_name="Promedios")

    print(f"Métricas combinadas y promediadas guardadas en {output_file}")


# ===============================
# Uso del Script
# ===============================
base_directory = "resultados_fl"  # Directorio donde están las métricas
output_excel_file = "metricas_combinadas.xlsx"  # Nombre del archivo Excel resultante

# Ejecutar la combinación de métricas
combinar_metricas_en_excel(base_directory, output_excel_file)
