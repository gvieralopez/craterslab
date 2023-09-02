# Craterslab Example Script No. 7:
# Advanced profile analysis. Computing and plotting crater slopes

from craterslab.sensors import DepthMap
from craterslab.visuals import plot_2D, plot_3D, plot_profile
from craterslab.ellipse import EllipticalModel


# depth_map = DepthMap.load('examples/data/fluidized_1.npz')
depth_map = DepthMap.load('examples/fluidized_1.npz')
depth_map.auto_crop()

em = EllipticalModel(depth_map, 20)
p = em.max_profile()
m1, m2 = p.slopes()

print(f'Computed slopes {m1=} {m2=}')

plot_3D(depth_map, profile=p, ellipse=em, preview_scale=(1, 1, 4))
plot_2D(depth_map, profile=p, ellipse=em)
plot_profile(p, draw_slopes=True, block=True)
