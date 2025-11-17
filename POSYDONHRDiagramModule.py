import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pandas as pd

from pathlib import Path
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColorBar, LinearColorMapper, LogColorMapper, ColumnDataSource, LogAxis, HoverTool
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import RdYlBu11
from bokeh.palettes import interp_palette
from bokeh.io import output_notebook
from bokeh.embed import file_html
from bokeh.resources import CDN

import inspect
import os
import webbrowser

def loadEndStateData(dataset, path =''):
    filterdf = pd.DataFrame()

    chunksize = 10 ** 6
    if path == '':
        for chunk in pd.read_hdf(dataset, key = 'history', chunksize=chunksize):
            # chunk is a DataFrame. To "process" the rows in the chunk:
            endCon = chunk['event'] == 'END'
            stateCon = chunk['state'] != 'merged' 
            filtChunk = chunk[endCon & stateCon]

            filterdf = pd.concat([filterdf, filtChunk])
        return filterdf
    else:
        for chunk in pd.read_hdf(path / dataset, key = 'history', chunksize=chunksize):
            # chunk is a DataFrame. To "process" the rows in the chunk:
            endCon = chunk['event'] == 'END'
            stateCon = chunk['state'] != 'merged' 
            filtChunk = chunk[endCon & stateCon]

            filterdf = pd.concat([filterdf, filtChunk])
        return filterdf
    
def loadZamsData(dataset, path =''):
    filterdf = pd.DataFrame()

    chunksize = 10 ** 6
    if path == '':
        for chunk in pd.read_hdf(dataset, key = 'history', chunksize=chunksize):
            # chunk is a DataFrame. To "process" the rows in the chunk:
            endCon = chunk['event'] == 'ZAMS'
            stateCon = chunk['state'] != 'merged' 
            filtChunk = chunk[endCon & stateCon]

            filterdf = pd.concat([filterdf, filtChunk])
        return filterdf
    else:
        for chunk in pd.read_hdf(path / dataset, key = 'history', chunksize=chunksize):
            # chunk is a DataFrame. To "process" the rows in the chunk:
            endCon = chunk['event'] == 'ZAMS'
            stateCon = chunk['state'] != 'merged' 
            filtChunk = chunk[endCon & stateCon]

            filterdf = pd.concat([filterdf, filtChunk])
        return filterdf

    
# function for making HR diagrams using POSYDON data
def HR_Diagram     (df,  # Pandas dataframe to used (or H5). however, it is reccomeneded to load the df into memory first to reduce reloading it everytime
                    df_name, # name of the Dataframe, this will be used for automatic title and filename generation.
                    history = True,
                    init_or_final = 'final',
                    path = '',
                    Star = 2, # which star, 1 or 2, of the POSYDON df to graph

                    variable='S1_mass',  # variable to be used on the colorbar
                    var_name='default',  # name of the colorbar var
                    LogVar = False,  # whether or not to Log10 the var used for the colorbar. for example, one would need to log10 orbital period in order for it to be readable

                    title='default',  # title of graph. if let to default it will automatically populate one based on input conditions
                    saveLoc='',  # filepath save location of graph
                    saveGraph = False
                    Star_Radius = True,  # T/F. If set to true automatically calculates the size of the graphed point based off of the radius of the star. If NOT set to True, must input a val for the star radius ex. Star_Radius = 4
                    fileName='Default',  # filename, if left to default one is autogened based on graphing vars
                    showGraph= True, # whether or not to output the graph inline. this is very useful to disable when generating repetable graphs that are being used for something else, as it prevents bloat. for example, figs for a LaTeX paper. 
                    palette = 'Default', #what pallete to use. this can be useful in somes cases where the plotted stars have colors similar to the white BG. 
                    style = 'default', # what graph style to use from the PLT selection
                    dpi = 200, # dpi/res to use

                    # y-limits
                    # this may not be needed entirely? very few cases where you'd want y-lims and wouldnt just be focused on the loaded df
                    ylimit=False,  # whether or not to use a set range or automatically fit one 
                    minR=1.5,  # min Y limit on graph
                    maxR=6.5,  # max y limit on graph

                    # reference star graphing logic
                    referenceStar=False,  # T/F, whether or not to reference point to graph
                    referenceStarName = 'Reference Star', # name of the reference star. ex. v404 Cygni 
                    referenceStarRange=False,  # T/F whether or not to graph an error range for the reference point
                    exampleLum=0,  # ref points lum. NOT LOGGED
                    exampleTemp=0,  # ref points temp. NOT LOGGED
                    exampleTempMin=0,
                    exampleTempMax=0,
                    exampleLumMin=0,
                    exampleLumMax=0, 
                    grapher = 'plt'):

    if history == True:
        S2_log_R = 'S2_log_R'
        S2_log_L = 'S2_log_L'
        S1_log_R = 'S1_log_R'
        S1_log_L = 'S1_log_L'
    elif init_or_final == 'init':
        S2_log_R = 'S2_log_R_i'
        S2_log_L = 'S2_log_L_i'
        S1_log_R = 'S1_log_R_i'
        S1_log_L = 'S1_log_L_i'

        variable = 'S1_mass_i'
    elif init_or_final == 'final':
        S2_log_L = 'S2_log_L_f'
        S1_log_R = 'S1_log_R_f'
        S1_log_L = 'S1_log_L_f'
        S2_log_R = 'S2_log_R_f'

        variable = 'S1_mass_f'
    else:
        print('not a valiv option for initOrFinal! options or "init" or "final"')


    if grapher == 'plt':
    
        plt.style.use(style) #graph style
        fig, ax = plt.subplots(figsize = (8,8))  # create figure and axis
        ax.grid(True)  # turn on grid
        ax.set_axisbelow(True)  # make grid lines draw below plotted points. totally an aesthetic choice
        ax.yaxis.grid(color='gray', linestyle='dashed')  # customize grid style
        
        cm = plt.colormaps['RdYlBu']  #This is the color map for the stars
        
        if Star == 2: # this logic is kinda horrid, revisit probably. and doesnt work! 
            if Star_Radius == True:
                r_dot = 10 ** df[S2_log_R]
            else:
                r_dot = Star_Radius

            # assings axis
            Temp = np.log10((((10 ** df[S2_log_L])/(10 ** df[S2_log_L])**2)**.25) * 5772)
            Lum = df[S2_log_L]
        else: 
            if Star_Radius == True:
                r_dot = 10 ** df[S1_log_R]
            else:
                r_dot = Star_Radius

            # assings axis
            Temp = np.log10((((10 ** df[S1_log_L])/(10 ** df[S1_log_L])**2)**.25) * 5772)
            Lum = df[S1_log_L]

        # binds the color of the scatter points to the x location (temp) of the star
        if LogVar == True:
            c = np.log10(df[variable])
        else:
            c = (df[variable])
        colors = c

        # labels
        ax.set_xlabel(r'log$_{10}$ Temperature [K]')
        ax.set_ylabel(r'log$_{10}$ Luminosity [$L_{\odot}$]')
        
        if ylimit == True:
            ax.set_ylim(minR, maxR)

        #scatter points. "cmap" is setting the colormap to use, "c" is setting the color itself (based on location), "s" is setting the size of the dot based off of star radii
        scatter = ax.scatter(Temp, Lum, cmap = cm, c = colors, s = r_dot)

        ax.invert_xaxis()

        # color bar stuff
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')  # <-- link colorbar to that scatter
        if LogVar == False:
            cbar.set_label(var_name if var_name != 'default' else variable) 
        else: 
            cbar.set_label(r'log$_{10}$ '+ var_name)
        
        #cbar.ax.invert_xaxis() #invert the color bar  (to match the inverted x scaling)
        #cbar.set_ticks([np.min(c),numpy.median,np.max(c),]) # remove the annoying ticks and labels

        if referenceStar == True:
            scatter = ax.scatter(np.log10(exampleTemp), np.log10(exampleLum),
            color = 'black', s = 100, marker = '*')
            if referenceStarRange == True:

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

        # Smart title if default
        if title == 'default':
            title = f"HR Diagram of {df_name} colored by {'log₁₀ ' if LogVar == True else ''}{var_name if var_name != 'default' else variable} with {'relative' if Star_Radius == True else f'radius={Star_Radius}'} point size"
        ax.set_title(title)
        # Smart filename if default
        if fileName == 'Default':
            file_parts = [
                df_name,
                variable,
                'log10' if LogVar == True else 'linear',
                str(Star_Radius) if Star_Radius != True else 'dynR'
            ]
            fileName = '_'.join(file_parts) + '.png'
            fileName = fileName.replace(" ", "_")
        else:
            fileName = fileName if fileName.endswith('.png') else f"{fileName}.png"
        
        if saveGraph == True
            # Save the figure
            save_path = Path(saveLoc) / fileName
            plt.savefig(save_path, dpi=dpi)
            plt.style.use('default')

        if showGraph== True:
            plt.show()
        plt.close()



    ### Bokeh Graphing
    elif grapher == 'bokeh':
            # whether or not to output the graph to be inline
        if showGraph == True: 
            output_notebook()

        # Build custom smooth palette. this is because the RdYlBu11 pallet by default only has 11 values. 
        if palette == 'Default':
            smooth_palette = interp_palette(RdYlBu11[::-1], 256)
        else:
            smooth_palette = palette

        # generating temp and lum data based on if star 1 or 2 is being used. note this does not have proper protection logic currently and needs a way of dropping NaN values
        if Star == 2:
            if Star_Radius == True:
                r_dot= 4 ** df['S2_log_R'] + 1 # get star radius from df. the +1 is to prevent radius values of 0... which is fine cause its all relative anyway
            else:
                if isinstance(Star_Radius, int): # check if Star_Radius is a float, and if so, set the r_dot to it. 
                    r_dot = float(Star_Radius) 
                else:
                    print("Star_Radius must be an int!")

            # Assign temp and lum Value

            Temp = np.log10( #rearranged Stefan-Boltzmann equation for calculating Temp based off of lum and radius
                    (
                        (10 ** df[S2_log_L]) /
                        (10 ** df[S2_log_R]) ** 2
                    ) ** 0.25
                    * 5772
                ) 
            Lum = df[S2_log_L]

        else: 
            if Star_Radius == True:
                r_dot= 4 ** df[S1_log_R] + 1
            else:
                r_dot = float(Star_Radius)

            # Axis values
            Temp = np.log10( #rearranged Stefan-Boltzmann equation for calculating Temp based off of lum and radius
                    (
                        (10 ** df[S2_log_L]) /
                        (10 ** df[S2_log_R]) ** 2
                    ) ** 0.25
                    * 5772
                ) 
            Lum = df[S1_log_L]

        # Smart title gen. sorta rough to look at, but simple when boken down
        if title == 'default':
            title = f"HR Diagram of {df_name} colored by {'Log$$_{10}$$ ' if LogVar == True else ''}{var_name if var_name != 'default' else variable} with {'relative' if Star_Radius == True else f'radius={Star_Radius}'} point size "
        
        # check if colorVar is already logged to prevent "double logging"
        if variable.startswith('lg'):
            LogVar = True

        # Color mapping    
        if LogVar == True: # yes this is slightly redundent, but this really makes sure that colordata isnt logged twice and is properly graphed
            if variable.startswith('lg'):
                color_data = 10 ** df[variable] #delog logged var for the LogColorMapper 🫠. this sounds insane, but this makes sure the colorbar shows values in proper 10^n format
            else:
                color_data = (df[variable])
            color_label = f"Log$${10}$$ {var_name}"
            mapper = LogColorMapper(palette=smooth_palette, low= np.min(color_data), high=np.nanmax(color_data))

        else:
            color_data = df[variable]
            if var_name != 'default':
                color_label = var_name
            else: color_label = variable
            mapper = LinearColorMapper(palette=smooth_palette, low=np.nanmin(color_data), high=np.nanmax(color_data))


        # Ensure all inputs are valid lists
        Temp_list = Temp.tolist()
        Lum_list = Lum.tolist()
        color_val_list = color_data.tolist()

        # r_dot might be a constant if Star_Radius is not True
        if isinstance(r_dot, (pd.Series, np.ndarray)):
            size_list = r_dot.tolist()
        else:
            size_list = [r_dot] * len(df)

        # Build the source dict for the figure. this includes both the standerd x/y cords and color vals
        source = ColumnDataSource(data=dict(
            Temp=Temp_list, # "X Value"
            Lum=Lum_list, # "Y-value"
            size=size_list, # size of point
            color_val=color_val_list, # color value
            log_Temp=np.round(Temp_list, 2), # temp hovor value
            log_Lum=np.round(Lum_list, 2), # Lum hover value
            var_val=np.round(color_val_list, 2) # variable hover value 
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

        if ylimit == True:
            p.y_range.start = minR
            p.y_range.end = maxR

        # all stuff to make the hover tools work
        hover = HoverTool(
            tooltips=[
                ("log₁₀ Temp", "@log_Temp"),
                ("log₁₀ Lum", "@log_Lum"),
                (f"{variable}", "@var_val")
            ]
        )
        p.add_tools(hover)

        ref_hover = HoverTool(
            tooltips=[
                ("Reference Star", "@label"),
                ("log₁₀ Temp", "@Temp"),
                ("log₁₀ Luminosity", "@Lum")
            ], 
            renderers=[]
        )
        p.add_tools(ref_hover)
        
        # Scatter plot with color and size
        p.scatter(
            x='Temp', y='Lum', source=source,
            size='size', marker='circle',
            fill_color={'field': 'color_val', 'transform': mapper},
            fill_alpha=0.8, line_color=None
        )


        # Invert X axis. i love how easy this is in bokeh 🙏
        p.x_range.flipped = True

        # Add color bar and put it on the right
        color_bar = ColorBar(color_mapper=mapper, label_standoff=12, title=color_label, location=(0, 0))
        p.add_layout(color_bar, 'right')

        #example star source data
        exampleSource = ColumnDataSource(data=dict(
            Temp=[np.log10(exampleTemp)],
            Lum=[np.log10(exampleLum)],
            label=[referenceStarName]
        ))

        # Reference star point marker and range box
        if referenceStar == True:
            star = p.scatter(
                x='Temp',
                y='Lum',
                source=exampleSource,
                size=15,
                marker='star',
                color='black',
                legend_label=referenceStarName
            )
            ref_hover.renderers = [star]

            # rect range of values of the star
            if referenceStarRange == True:
                width = np.log10(exampleTempMax) - np.log10(exampleTempMin)
                height = np.log10(exampleLumMax) - np.log10(exampleLumMin)

                # example box source. this has some extra logic so when hovering over it it gives useful info
                rect_source = ColumnDataSource(data=dict(
                    x=[np.log10(exampleTempMin) + width / 2],
                    y=[np.log10(exampleLumMin) + height / 2],
                    width=[np.log10(exampleTempMax) - np.log10(exampleTempMin)],
                    height=[np.log10(exampleLumMax) - np.log10(exampleLumMin)],
                    label=["Error Range"],
                    T_min=[np.log10(exampleTempMin)],
                    T_max=[np.log10(exampleTempMax)],
                    L_min=[np.log10(exampleLumMin)],
                    L_max=[np.log10(exampleLumMax)]
                ))

                rect = p.rect(
                    x='x', y='y',
                    width='width', height='height',
                    source=rect_source,
                    fill_alpha=0.2,
                    fill_color='gray',
                    line_color='black'
                )
                
                range_hover = HoverTool(
                    tooltips=[
                        ("🞄", "@label"),
                        ("log₁₀ Temp range", "@T_min → @T_max"),
                        ("log₁₀ Lum range", "@L_min → @L_max")
                    ],
                    renderers=[rect]
                )
                p.add_tools(range_hover)
                p.add_tools(ref_hover) # this seems janky, and maybe it kinda is, but it prevents the error range box from drawing over the ref star hover box. some draw order stuff

        # Auto-generate filename if left to default
        if fileName == 'Default':
            file_parts = [
                df_name,
                variable,
                'log10' if LogVar == True else 'linear',
                'dynR' if Star_Radius == True else f'R{int(Star_Radius)}'
            ]
            fileName = '_'.join(file_parts).replace(" ", "_") + '.html'

        fileName = fileName if fileName.endswith('.html') else f"{fileName}.html"
        
        # FIXED!!
        # some great funky logic to allow for stuff to work, typically one would call output_file(savepath, blah blah blah), but that permantly overwrites the functionality of show(p), which sucks. the save function prevent this.
        # are you kidding me?? this logic works on windows totally fine (least on my win10 machine) but on linux doesnt??
        # 
        # save_path = Path(saveLoc) / fileName
        # save(p, filename=str(save_path), title=title)
        
        # if showGraph == True:
        #     show(p)
        # FIXED!!
        # 
        # This is technically equivlent to whats above but iwht a little less control. For some bizzaro reason bokeh freaks out if you display in-line and try and save. probably doing something wrong, but hey, what is below works, thank you chatgpt (this might have some horrid side effect)
        
        if showGraph == True:
            show(p)

        html = file_html(p, CDN, title)
        save_path = Path(saveLoc) / fileName
        with open(save_path, "w") as f:
            f.write(html)
    else: 
        print('Not a valid grapher option! Options are plt for matplotlib and bokeh for bokeh')


def HR_Sampler (Database, 
                DataFrame_Name, 
                Star_R = True, 
                SaveLocation = 'default', 
                palette = 'Default', 
                Fopen = True):


    #file location saving logic
    if SaveLocation == 'default':
        GraphSaveLocation = Path().resolve() / 'Sampler' / DataFrame_Name.replace(" ", "_") / 'graphs'
        ViewerSaveLocation = Path().resolve() / 'Sampler' / DataFrame_Name.replace(" ", "_") / 'viewer.html'
    else:
        GraphSaveLocation = Path().resolve() / SaveLocation / 'graphs'
        ViewerSaveLocation = Path().resolve() / SaveLocation /'viewer.html'
 
    os.makedirs(GraphSaveLocation, exist_ok=True)

    # making the html viewer for the graphs
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

    <h2>X-ray Binaries HR Diagrams</h2>
    <select id="graphSelect">
    <option value="graphs/S1_mass.html">Star One Mass</option>
    <option value="graphs/S2_mass.html">Star Two Mass</option>
    <option value="graphs/eccentricity.html">Eccentricity</option>
    <option value="graphs/S2_surface_h1.html">Star Two Surface Hydrogen</option>
    <option value="graphs/S2_surface_he4.html">Star Two Surface Helium</option>
    <option value="graphs/lg_mtransfer_rate.html">Mass Transfer Rate (log10)</option>
    <option value="graphs/orbital_period.html">Orbital Period (log10)</option>
    </select>

    <label for="altCheckbox">Enable Dynamic Star Radius</label>
    <input type="checkbox" id="dynmRCheckbox">

    <iframe id="graphFrame" src=""></iframe>

    <script>
    const select = document.getElementById("graphSelect");
    const frame = document.getElementById("graphFrame");
    const dynmRCheckbox = document.getElementById("dynmRCheckbox");

    function updateFrame() {{
        const selectedGraph = select.value.replace("graphs/", "").replace(".html", "");
        const dynmRVersion = dynmRCheckbox.checked ? '_dynmR' : '';
        frame.src = `graphs/${{selectedGraph}}${{dynmRVersion}}.html`;
    }}

    select.addEventListener("change", updateFrame);
    dynmRCheckbox.addEventListener("change", updateFrame);
    updateFrame();
    </script>

    </body>
    </html>"""

    with open(ViewerSaveLocation, "w", encoding="utf-8") as file:
        file.write(ViewerHTML)
    
    graphValues =  {
        'df' : Database, 
        'df_name': DataFrame_Name,
        'LogVar' : False, 
        'title': 'default',  
        'saveLoc': GraphSaveLocation, 
        'Star_Radius' : Star_R,
        'palette' : palette,
        'showGraph':False,
        'grapher' :'bokeh'
        }

    HR_Diagram         (variable='S1_mass',  # variable to be used on the colorbar
                        var_name=r'Star One Mass \[M_{\odot}/y\]',  # name of the colorbar var
                        fileName='S1_mass',
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph 
                        Star_Radius = Star_R,

                        palette = palette,
                        showGraph=False,
                        grapher='bokeh'), # type: ignore
    
    HR_Diagram (variable='S1_mass',  # variable to be used on the colorbar
                var_name=r'Star One Mass \[M_{\odot}/y\]',  # name of the colorbar var
                fileName='S1_mass_dynmR',
                
                df = Database,  # database
                df_name=DataFrame_Name,
                LogVar = False,  # whether or not to Log10 the var used for the colorbar
                title='default',  # title of graph
                saveLoc=GraphSaveLocation,  # save location of graph

                palette = palette,
                showGraph=False,
                grapher='bokeh'), # type: ignore
                        

    HR_Diagram (variable='S2_mass',  # variable to be used on the colorbar
                        var_name=r'Star Two Mass \[M_{\odot}/y\]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius = Star_R,
                        fileName='S2_mass',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='S2_mass',  # variable to be used on the colorbar
                        var_name=r'Star Two Mass \[M_{\odot}/y\]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        fileName='S2_mass_dynmR',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='orbital_period',  # variable to be used on the colorbar
                        var_name=r'Orbital Period',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = True,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius = Star_R,
                        fileName='orbital_period',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='orbital_period',  # variable to be used on the colorbar
                        var_name=r'Orbital Period',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = True,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        fileName='orbital_period_dynmR',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='eccentricity',  # variable to be used on the colorbar
                        var_name=r'Eccentricity',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius= Star_R,
                        fileName='eccentricity',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='eccentricity',  # variable to be used on the colorbar
                        var_name=r'Eccentricity',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        fileName='eccentricity_dynmR',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='lg_mtransfer_rate',  # variable to be used on the colorbar
                        var_name=r'Mass Transfer Rate \[M_{\odot}/y\]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = True,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius= Star_R,
                        fileName='lg_mtransfer_rate',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='lg_mtransfer_rate',  # variable to be used on the colorbar
                        var_name=r'Mass Transfer Rate \[M_{\odot}/y\]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = True,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        fileName='lg_mtransfer_rate_dynmR',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='S2_surface_h1',  # variable to be used on the colorbar
                        var_name=r'Star Two Surface Hydrogen [%]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius = Star_R,
                        fileName='S2_surface_h1',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh'), # type: ignore
    
    HR_Diagram (variable='S2_surface_h1',  # variable to be used on the colorbar
                        var_name=r'Star Two Surface Hydrogen [%]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        fileName='S2_surface_h1_dynmR',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh'), # type: ignore
    
    HR_Diagram (variable='S2_surface_he4',  # variable to be used on the colorbar
                        var_name=r'Star Two Surface Helium [%]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        Star_Radius= Star_R,
                        fileName= 'S2_surface_he4',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    HR_Diagram (variable='S2_surface_he4',  # variable to be used on the colorbar
                        var_name=r'Star Two Surface Helium [%]',  # name of the colorbar var
                        df = Database,  # database
                        df_name=DataFrame_Name,
                        LogVar = False,  # whether or not to Log10 the var used for the colorbar
                        title='default',  # title of graph
                        saveLoc=GraphSaveLocation,  # save location of graph
                        fileName= 'S2_surface_he4_dynmR',
                        palette = palette,
                        showGraph=False,
                        grapher='bokeh')
    
    if Fopen == True:
        webbrowser.open(ViewerSaveLocation.resolve().as_uri())

