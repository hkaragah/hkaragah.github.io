import bpy
import math
import os


# --- Cleanup ---
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# --- Units & Parameters ---
FT_TO_M = 0.3048
IN_TO_M = 0.0254

height_ft = 14
column_height = height_ft * FT_TO_M  # 28 ft in meters

# W14x48 section dimensions (in inches → meters)
d = 13.8 * IN_TO_M   # overall depth
bf = 8.03 * IN_TO_M   # flange width
tw = 0.34 * IN_TO_M   # web thickness
tf = 0.595 * IN_TO_M   # flange thickness
length = column_height  # extrusion length

# --- Create 2D Cross Section of I-Beam ---
def create_I_beam_cross_section():
    verts = []
    faces = []

    x = bf / 2
    y = d / 2
    
    # Start at bottom left corner, go clockwise
    verts = [
        (-x, -y), #1
        (-x, -y + tf), #2
        (-tw/2, -y + tf), #3 
        (-tw/2,  y - tf), #4
        (-x, y - tf), #5
        (-x, y), #6
        ( x, y), #7
        ( x, y - tf), #8
        ( tw/2,  y - tf), #9
        ( tw/2, -y + tf), #10
        ( x, -y + tf), #11
        ( x, -y) #12
    ]
    faces = [list(range(len(verts)))]

    mesh = bpy.data.meshes.new("IBeamCrossSection")
    obj = bpy.data.objects.new("IBeamCrossSection", mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata([(x, y, 0) for x, y in verts], [], faces)
    mesh.update()

    return obj

# --- Ground and shadow ---
def add_ground_and_shadow():
    # Delete existing "Ground" if it exists
    if "Ground" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Ground"], do_unlink=True)

    # Create round ground (cylinder)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=1.5,
        depth=0.1,
        location=(0, 0, -0.05)  # Slightly below Z = 0
    )
    ground = bpy.context.active_object
    ground.name = "Ground"

    # Assign a diffuse material to ground
    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
    ground.data.materials.append(mat)

    # Create a sun lamp for shadows
    bpy.ops.object.light_add(type='SUN', location=(10, -10, 20))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 10
    sun.rotation_euler = (math.radians(-60), math.radians(0), math.radians(60))
    sun.data.shadow_soft_size = 1.0
    sun.data.use_shadow = True

    # Set render engine to Cycles (or EEVEE if preferred)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'  # Optional: switch to CPU if needed

    # Optional: enable ambient occlusion and soft shadows
    bpy.context.scene.eevee.use_soft_shadows = True
    bpy.context.scene.eevee.use_gtao = True

    # Make sure shadow settings are enabled for the ground and other objects
    ground.display_type = 'SOLID'  # Make sure it's solid in the viewport
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.visible_shadow = True  # Enable shadow visibility for mesh objects

    return ground


# --- Extrude the 2D shape ---
def extrude_shape(obj, length):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.convert(target='MESH')
    
    # Toggle to edit mode to extrude the shape
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, length)}  # Extrude in Z-direction
    )
    bpy.ops.object.editmode_toggle()

    # Rename the object
    obj.name = "W14x48_Column"
    obj.location = (0, 0, 0)

    # Create a steel red material
    steel_red = bpy.data.materials.new(name="SteelRedMat")
    steel_red.use_nodes = True
    bsdf = steel_red.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.8, 0.0, 0.0, 1)  # RGB for steel red

    # Assign the material to the object
    if len(obj.data.materials) > 0:
        obj.data.materials[0] = steel_red
    else:
        obj.data.materials.append(steel_red)

    return obj

# --- Create Arrows ---
def create_arrow(name, location, rotation=(0, 0, 0), color=(1, 0, 0, 1), shaft_length=0.5, head_length=0.3):
    import bpy
    import mathutils

    # Create cone (arrowhead) with tip at Z=0
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.05,
        depth=head_length,
        location=(0, 0, head_length / 2)  # moves tip to origin
    )
    head = bpy.context.active_object
    head.name = name + "_head"

    # Create cylinder (shaft) below cone
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.02,
        depth=shaft_length,
        location=(0, 0, -shaft_length / 2)  # below origin
    )
    shaft = bpy.context.active_object
    shaft.name = name + "_shaft"

    # Join them into one object
    bpy.ops.object.select_all(action='DESELECT')
    head.select_set(True)
    shaft.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.join()
    arrow = head
    arrow.name = name

    # Set origin to tip of arrowhead (Z=0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    # Move cursor to world origin before setting origin (so Z=0 becomes origin)
    bpy.context.scene.cursor.location = (0, 0, head_length)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    # Now that origin is at tip, we can rotate then move it to location
    arrow.rotation_euler = rotation
    arrow.location = location

    # Add color
    mat = bpy.data.materials.new(name + "_Mat")
    mat.diffuse_color = color
    if len(arrow.data.materials) == 0:
        arrow.data.materials.append(mat)
    else:
        arrow.data.materials[0] = mat

    return arrow



# --- Create Labels ---
def create_label(text, location, rotation=(0, 0, 0), color=(1, 0, 0, 1)):
    bpy.ops.object.text_add(location=location, align='VIEW', rotation=rotation)
    txt = bpy.context.active_object
    txt.data.body = text
    txt.scale = (0.6, 0.6, 0.6)

    # Add color
    mat = bpy.data.materials.new("_Mat")
    mat.diffuse_color = color
    if len(txt.data.materials) == 0:
        txt.data.materials.append(mat)
    else:
        txt.data.materials[0] = mat
        
    return txt

# --- Build the model ---
cross_section = create_I_beam_cross_section()
column = extrude_shape(cross_section, column_height)

# Move so base is at z = 0
column.location = (0, 0, 0)
add_ground_and_shadow()

# --- Add force arrows at the top ---
top_z = column_height
create_arrow("GravityArrow", location=(0, 0, column_height), rotation=(math.radians(180), 0, 0))
create_label("200 kip", location=(0, -0.25, column_height + 0.25), rotation=(math.radians(90), 0, math.radians(-90)), color=(1, 0, 0, 1))

create_arrow("LateralArrow", location=(0, d/2, column_height), rotation=(math.radians(90), 0, 0), color=(0, 0, 1, 1))
create_label("1 kip", location=(0, 1.5, column_height - 0.75), rotation=(math.radians(90), 0, math.radians(-90)), color=(0, 0, 1, 1))


# Check if there's already a camera
if "Camera" not in bpy.data.objects:
    # Create a new camera
    bpy.ops.object.camera_add(location=(-10, 10, 14), rotation=(math.radians(50), 0, math.radians(-135)))
    camera = bpy.context.active_object
    camera.name = "Camera"
else:
    camera = bpy.data.objects["Camera"]

# Set the camera as the active camera
bpy.context.scene.camera = camera

# Set rendering engine to 'BLENDER_EEVEE' (OpenGL render requires it)
bpy.context.scene.render.engine = 'BLENDER_EEVEE'

# Enable transparent background
bpy.context.scene.render.film_transparent = True

# Get path of the current .blend file
blend_dir = bpy.path.abspath("//")
filename = "cant_col_p_delta_rendered_image.png"

# Combine the path and filename
output_path = os.path.join(blend_dir, filename)

# Set the resolution (you can adjust these values for higher or lower resolution)
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# Set the file format to PNG
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Enable RGBA for alpha channel
bpy.context.scene.render.image_settings.color_mode = 'RGBA'

# Set the output path for the image
bpy.context.scene.render.filepath = output_path

# Perform the rendering
bpy.ops.render.render(write_still=True)

# Notify that the render is done
print(f"Render saved to: {output_path}")

## Render the active viewport (as seen on screen)
#bpy.ops.render.opengl(write_still=True)

#print(f"Viewport render saved to: {output_path}")