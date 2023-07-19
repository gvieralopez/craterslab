from craterslab.sensors import DepthMap, SensorResolution
from craterslab.visuals import plot_2D, plot_3D, plot_profile
from craterslab.craters import Surface
from craterslab.ellipse import EllipticalModel
import pandas as pd

# Define sensor resolution
data_resolution = SensorResolution(235.65, 235.65, 1.0, "m")

# Define data sources
depth_map = DepthMap.from_xyz_file(
    "king.xyz", resolution=data_resolution, rescaled_with=(1, 1, 1)
)
depth_map.crop_borders(ratio=0.25)

s = Surface(depth_map)
p = s.max_profile

plot_3D(depth_map, profile=s.max_profile, ellipse=s.em, preview_scale=(1, 1, 5))
plot_2D(depth_map, profile=s.max_profile, ellipse=s.em)
plot_profile(s.max_profile, block=True)

print(s)

# Obtener los valores de s y h
# s_values = p.s
# h_values = p.h

# # Crear un DataFrame con los valores de s y h
# data = {'Distancia': s_values, 'Profundidad': h_values}
# df = pd.DataFrame(data)

# # Especificar el nombre del archivo Excel de salida
# nombre_archivo_excel = 'perfil_profundidad.xlsx'

# # Guardar el DataFrame en el archivo Excel
# df.to_excel(nombre_archivo_excel, index=False)

# print(f"Los valores se han exportado exitosamente a '{nombre_archivo_excel}'.")



