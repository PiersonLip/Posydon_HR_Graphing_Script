import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from pathlib import Path

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColorBar, LinearColorMapper, LogColorMapper, ColumnDataSource, LogAxis
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import RdYlBu11
import numpy as np
import pandas as pd
from pathlib import Path

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColorBar, LinearColorMapper, LogColorMapper, ColumnDataSource, LogAxis, HoverTool
from bokeh.transform import linear_cmap, log_cmap
import numpy as np
import pandas as pd
from pathlib import Path
from bokeh.palettes import interp_palette



import inspect

import os

import webbrowser

# function for making HR diagrams using POSYDON data

def get_caller_var_name(var):
    frame = inspect.currentframe().f_back
    for name, val in frame.f_locals.items():
        if id(val) == id(var):
            return name
    return "DB"

def color_map_HR (DB, # database
                  variable = 'S1_mass', # variable to be used on the colorbar
                  var_name = 'Star One Mass', # name of the colorbar var
                  LogVar = 'F', # whether or not to Log10 the var used for the colarbar 
                  title = 'default', #title of graph
                  saveLoc = '', #save location of graph
                  examplePoint = 'F', # T/F, whether or not to reference point to graph
                  examplePointRange = 'F', # T/F whether or not to graph an error range for the reference point
                  exampleLum = 0, # ref points lum. NOT LOGGED
                  exampleTemp = 0, # ref points temp. NOT LOGGED
                  exampleTempMin = 0, 
                  exampleTempMax = 0, 
                  exampleLumMin =0, 
                  exampleLumMax = 0, 
                  Star_Radius = 'T',  # true OR VALUE. If not set to T, must input a val for the star radius whether or not to automatically use the star reference
                  ylimit = 'T', # whether or not to use a set range or autogen
                  minR = 1.5, # minimum Y limit
                  maxR = 6.5, # max y val
                  style = 'default', #graph style
                  fileName ='Default', #filename, if set to default one is autogened
                  dpi = 200): # graph res
    
    plt.style.use(style) #graph style
    fig, ax = plt.subplots(figsize = (8,8))  # create figure and axis
    ax.grid(True)  # turn on grid
    ax.set_axisbelow(True)  # make grid lines draw below plotted points
    ax.yaxis.grid(color='gray', linestyle='dashed')  # customize grid style
    
    cm = plt.colormaps['RdYlBu']  #This is the color map for the stars
    
    if Star_Radius == 'T':
        r_dot = 10 ** DB['S2_log_R']
        # plt.suptitle("Size of dot corresponds to Donor radius", fontsize=10, family="monospace", color='.5')
    else:
        r_dot = Star_Radius

    # assings axis
    Temp = np.log10((((10 ** DB['S2_log_L'])/(10 ** DB['S2_log_R'])**2)**.25) * 5772)
    Lum = DB['S2_log_L']

    # binds the color of the scatter points to the x location (temp) of the star
    if LogVar == 'T':
        c = np.log10(DB[variable])
    else:
        c = (DB[variable])
    colors = c

    # labels
    ax.set_xlabel(r'$log_{10}$ Temperature [K]')
    ax.set_ylabel(r'$log_{10}$ Luminosity [$L_{\odot}$]')
    
    if ylimit == 'T':
        ax.set_ylim(minR, maxR)

    #scatter points. "cmap" is setting the colormap to use, "c" is setting the color itself (based on location), "s" is setting the size of the dot based off of star radii
    scatter = ax.scatter(Temp, Lum, cmap = cm, c = colors, s = r_dot)

    ax.invert_xaxis()

    # color bar stuff
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')  # <-- link colorbar to that scatter
    if LogVar == 'F':
        cbar.set_label(var_name) 
    else: 
        cbar.set_label(r'log$_{10}$ '+ var_name)
    
    #cbar.ax.invert_xaxis() #invert the color bar  (to match the inverted x scaling)
    #cbar.set_ticks([np.min(c),numpy.median,np.max(c),]) # remove the annoying ticks and labels

    if examplePoint == 'T':
        scatter = ax.scatter(np.log10(exampleTemp), np.log10(exampleLum),
         color = 'black', s = 100, marker = '*')
        if examplePointRange == 'T':

            log_temp_min = np.log10(exampleTempMin)
            log_temp_max = np.log10(exampleTempMax)
            log_lum_min = np.log10(exampleLumMin)
            log_lum_max = np.log10(exampleLumMax)

            # Draw the rectangle for the overlapping region
            width = log_temp_max - log_temp_min
            height = log_lum_max - log_lum_min

            rect = patches.Rectangle(
                (log_temp_min, log_lum_min), width, height,
                linewidth=1, edgecolor='Black', facecolor='gray', alpha=0.2
            )
            ax.add_patch(rect) # graphs a `patch`, looks better then error bars IMO


    # everything below is for auto building file default file names
    # Get short name of DB if possible
    db_name_short = get_caller_var_name(DB)

    # Determine Star_Radius string
    if Star_Radius == 'T':
        Star_Radius_STR = 'dynR'
    else:
        Star_Radius_STR = f'R{int(Star_Radius)}'

    # Smart title if default
    if title == 'default':
        log_mode = str(LogVar).strip().upper() == 'T'
        title = f"{db_name_short} HR Diagram colored by {'log₁₀ ' if log_mode else ''}{var_name} with point radius: {Star_Radius_STR}"
    ax.set_title(title)
    # Smart filename if default
    if fileName == 'Default':
        file_parts = [
            db_name_short,
            variable,
            'log10' if LogVar == 'T' else 'linear',
            Star_Radius_STR
        ]
        fileName = '_'.join(file_parts) + '.png'
        fileName = fileName.replace(" ", "_")
    else:
        fileName = fileName if fileName.endswith('.png') else f"{fileName}.png"

    # Save the figure
    save_path = Path(saveLoc) / fileName
    plt.savefig(save_path, dpi=dpi)
    plt.style.use('default')
    plt.close()



def color_map_HR_bokeh (DB,  # database
                        db_name="Default",
                        variable='S1_mass',  # variable to be used on the colorbar
                        var_name='default',  # name of the colorbar var
                        LogVar='F',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc='',  # save location of graph
                        examplePoint='F',  # T/F, whether or not to reference point to graph
                        examplePointRange='F',  # T/F whether or not to graph an error range for the reference point
                        exampleLum=0,  # ref points lum. NOT LOGGED
                        exampleTemp=0,  # ref points temp. NOT LOGGED
                        exampleTempMin=0,
                        exampleTempMax=0,
                        exampleLumMin=0,
                        exampleLumMax=0,
                        Star_Radius='T',  # true OR VALUE. If not set to T, must input a val for the star radius
                        ylimit='F',  # whether or not to use a set range or autogen
                        minR=1.5,  # minimum Y limit
                        maxR=6.5,  # max y val
                        fileName='Default',  # filename, if set to default one is autogened
                        open = 'F',
                        palette = 'Default'):  
    
    # Build custom smooth palette
    if palette == 'Default':
        smooth_palette = interp_palette(RdYlBu11[::-1], 256)
    else:
        smooth_palette = palette

    # Output filename generation
    if Star_Radius == 'T':
        r_dot= 4 ** DB['S2_log_R'] + 1
        Star_Radius_STR = 'star_radius'
    else:
        r_dot = float(Star_Radius)
        Star_Radius_STR = f"radius_{int(Star_Radius)}"

    # Axis values
    Temp = np.log10((((10 ** DB['S2_log_L']) / (10 ** DB['S2_log_R']) ** 2) ** 0.25) * 5772)
    Lum = DB['S2_log_L']

    # Smart title gen
    if title == 'default':
        title_parts = [
            f"HR Diagram colored by {'log₁₀ ' if LogVar == 'T' else ''}{var_name}",
            f"with {'relative' if Star_Radius == 'T' else f'radius={Star_Radius}'} point size"
        ]
        title = ", ".join(title_parts)

    # Color mapping
    color_data = DB[variable].replace([np.inf, -np.inf], np.nan).dropna()
    if LogVar == 'T':
        color_data = np.log10(DB[variable])
        color_label = f"log₁₀ {var_name}"
        mapper = LinearColorMapper(palette=smooth_palette, low= np.min(color_data), high=np.nanmax(color_data))

    else:
        color_data = DB[variable]
        color_label = var_name
        mapper = LinearColorMapper(palette=smooth_palette, low=np.nanmin(color_data), high=np.nanmax(color_data))


    # Ensure all inputs are valid lists for ColumnDataSource
    Temp_list = Temp.tolist() if hasattr(Temp, "tolist") else list(Temp)
    Lum_list = Lum.tolist() if hasattr(Lum, "tolist") else list(Lum)
    color_val_list = color_data.tolist() if hasattr(color_data, "tolist") else list(color_data)

    # r_dot might be a constant if Star_Radius is not 'T'
    if isinstance(r_dot, (pd.Series, np.ndarray)):
        size_list = r_dot.tolist()
    else:
        size_list = [r_dot] * len(DB)

    # Build the source dict
    source = ColumnDataSource(data=dict(
        Temp=Temp_list,
        Lum=Lum_list,
        size=size_list,
        color_val=color_val_list,
        log_Temp=np.round(Temp_list, 2),
        log_Lum=np.round(Lum_list, 2),
        var_val=np.round(color_val_list, 2)
    ))


    # Set up plot
    p = figure(
        width=700,
        height=700,
        title=title,
        x_axis_label=r'log\[_{10}\] Temperature [K]',
        y_axis_label=r'log\[_{10}\] Luminosity [$$L_{\odot}$$]',
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    if ylimit == 'T':
        p.y_range.start = minR
        p.y_range.end = maxR

    hover = HoverTool(
        tooltips=[
            ("log₁₀ Temp", "@log_Temp"),
            ("log₁₀ Lum", "@log_Lum"),
            (f"{var_name}", "@var_val")
        ]
    )
    p.add_tools(hover)
    
    # Scatter plot with color and size
    p.scatter(
        x='Temp', y='Lum', source=source,
        size='size', marker='circle',
        fill_color={'field': 'color_val', 'transform': mapper},
        fill_alpha=0.8, line_color=None
    )


    # Invert X axis 
    p.x_range.flipped = True

    # Add color bar
    color_bar = ColorBar(color_mapper=mapper, label_standoff=12, title=color_label, location=(0, 0))
    p.add_layout(color_bar, 'right')

    # Reference point marker and range box
    if examplePoint == 'T':
        p.star(
            x=[np.log10(exampleTemp)],
            y=[np.log10(exampleLum)],
            size=15, color='black', legend_label="Reference Star"
        )

        if examplePointRange == 'T':
            width = np.log10(exampleTempMax) - np.log10(exampleTempMin)
            height = np.log10(exampleLumMax) - np.log10(exampleLumMin)
            p.rect(
                x=np.log10(exampleTempMin) + width / 2,
                y=np.log10(exampleLumMin) + height / 2,
                width=width, height=height,
                fill_alpha=0.2, fill_color='gray', line_color='black'
            )

    # Auto-generate filename if default
    if fileName == 'Default':
        file_parts = [
            db_name,
            variable,
            'log10' if LogVar == 'T' else 'linear',
            'dynR' if Star_Radius == 'T' else f'R{int(Star_Radius)}'
        ]
        fileName = '_'.join(file_parts).replace(" ", "_") + '.html'

    fileName = fileName if fileName.endswith('.html') else f"{fileName}.html"
    
    save_path = Path(saveLoc) / fileName
    output_file(save_path)
    save(p)
    if open == 'T': 
        webbrowser.open(save_path.resolve().as_uri())

def HR_Diagram_Bokeh_Sample_Grapher(Database, DB_Name, Star_R = 'T', SaveLocation = 'default', palette = 'Default', Fopen = 'T'):
    databaseName = 'DB'

    #file location saving logic
    if SaveLocation == 'default':
        GraphSaveLocation = Path().resolve() / 'Sampler' / 'graphs'
        ViewerSaveLocation = Path().resolve() / 'Sampler' / 'viewer.html'
    else:
        GraphSaveLocation = Path().resolve() / 'Sampler' / SaveLocation  / 'graphs'
        ViewerSaveLocation = Path().resolve() / 'Sampler' / SaveLocation /'viewer.html'
 
    os.makedirs(GraphSaveLocation, exist_ok=True)

    #making the html viewer for the graphs
    ViewerHTML = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>Bokeh HR Viewer</title>
    <style>
        body {{ font-family: 'Times New Roman', Times, serif; padding: 5px; background-color: #f4f4f4; }}
        select {{ padding: 0.2em; font-size: 1.2em;}}
        iframe {{ width: 100%; height: 90vh; margin-top: 1em; border: 2px solid #aaa; background: #fff; }}
    </style>
    </head>
    <body>

    <h2>{DB_Name} HR Diagrams</h2>
    <select id="graphSelect">
    <option value="graphs/S1_mass.html">Star One Mass</option>
    <option value="graphs/S2_mass.html">Star Two Mass</option>
    <option value="graphs/eccentricity.html">Eccentricity</option>
    <option value="graphs/S2_surface_h1.html">Star Two Surface Hydrogen</option>
    <option value="graphs/S2_surface_he4.html">Star Two Surface Helium</option>
    <option value="graphs/lg_mtransfer_rate.html">Mass Transfer Rate (log10) </option>
    <option value="graphs/orbital_period.html">Orbital Period (log10) </option>

    </select>

    <iframe id="graphFrame" src=""></iframe>

    <script>
    const select = document.getElementById("graphSelect");
    const frame = document.getElementById("graphFrame");

    function updateFrame() {{
        frame.src = select.value;
    }}

    select.addEventListener("change", updateFrame);
    updateFrame(); // auto-load first option
    </script>

    </body>
    </html>"""

    with open(ViewerSaveLocation, "w", encoding="utf-8") as file:
        file.write(ViewerHTML)
    

    color_map_HR_bokeh (variable='S1_mass',  # variable to be used on the colorbar
                        var_name=r'Star One Mass [$$M_{\odot}$$]',  # name of the colorbar var
                        DB = Database,  # database
                        db_name=databaseName,
                        LogVar='F',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius = Star_R,
                        fileName='S1_Mass',
                        palette = palette),
                        
    
    color_map_HR_bokeh (variable='S2_mass',  # variable to be used on the colorbar
                        var_name=r'Star Two Mass [$$M_{\odot}$$]',  # name of the colorbar var
                        DB = Database,  # database
                        db_name=databaseName,
                        LogVar='F',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius = Star_R,
                        fileName='S2_Mass',
                        palette = palette)
    
    color_map_HR_bokeh (variable='orbital_period',  # variable to be used on the colorbar
                        var_name=r'Orbital Period',  # name of the colorbar var
                        DB = Database,  # database
                        db_name=databaseName,
                        LogVar='T',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius= Star_R,
                        fileName='Orbital_Period',
                        palette = palette)
    
    color_map_HR_bokeh (variable='eccentricity',  # variable to be used on the colorbar
                        var_name=r'Eccentricity',  # name of the colorbar var
                        DB = Database,  # database
                        db_name=databaseName,
                        LogVar='F',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius= Star_R,
                        fileName='Eccentricity',
                        palette = palette)
    
    color_map_HR_bokeh (variable='lg_mtransfer_rate',  # variable to be used on the colorbar
                        var_name=r'Log\[_{10}\] Mass Transfer Rate \[M_{\odot}/y\]',  # name of the colorbar var
                        DB = Database,  # database
                        db_name=databaseName,
                        LogVar='F',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius= Star_R,
                        fileName='lg_mtransfer_rate',
                        palette = palette)
    
    color_map_HR_bokeh (variable='S2_surface_h1',  # variable to be used on the colorbar
                        var_name=r'Star Two Surface Hydrogen [%]',  # name of the colorbar var
                        DB = Database,  # database
                        db_name=databaseName,
                        LogVar='F',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius = Star_R,
                        fileName='S2_surface_h1',
                        palette = palette),
    
    color_map_HR_bokeh (variable='S2_surface_he4',  # variable to be used on the colorbar
                        var_name=r'Star Two Surface Helium [%]',  # name of the colorbar var
                        DB = Database,  # database
                        db_name=databaseName,
                        LogVar='F',  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius= Star_R,
                        fileName= 'S2_surface_he4',
                        palette = palette)
    
    # color_map_HR_bokeh (variable='S1_spin',  # variable to be used on the colorbar
    #                     var_name='Star One Spin Rate',  # name of the colorbar var
    #                     DB = Database,  # database
    #                     db_name=databaseName,
    #                     LogVar='T',  # whether or not to Log10 the var used for the colorbar
    #                     title='default',  # title of graph
    #                     saveLoc=GraphSaveLocation,  # save location of graph
    #                     Star_Radius= Star_R)
    
    # color_map_HR_bokeh (variable='S2_spin',  # variable to be used on the colorbar
    #                     var_name='Star Two Spin Rate',  # name of the colorbar var
    #                     DB = Database,  # database
    #                     db_name=databaseName,
    #                     LogVar='T',  # whether or not to Log10 the var used for the colorbar
    #                     title='default',  # title of graph
    #                     saveLoc=GraphSaveLocation,  # save location of graph
    #                     Star_Radius= Star_R)

    if Fopen == 'T':
        webbrowser.open(ViewerSaveLocation.resolve().as_uri())
