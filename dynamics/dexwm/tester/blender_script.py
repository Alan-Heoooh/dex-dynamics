import bpy

obj = bpy.context.active_object

mesh = obj.data
selected_indices = [i.index for i in mesh.vertices if i.select]

print(selected_indices)
