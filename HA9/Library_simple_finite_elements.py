import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import FuncFormatter

def lokale_rotationsmatrix(phi):
    """
    Parameters
    ----------
    phi : Winkel in Grad
        Mathematisch positiver Rotationswinkel um den die Stange aus ihrer 
        horizontalen Lage um ihren linken Knoten rotiert wurde.

    Returns
    -------
    R : Entsprechende 6x6 Rotationsmatrix zur Koordinatentransformation
        
    """
    # Achtung: phi in Grad, wird hier umgewandelt
    phi = np.deg2rad(phi)
    # 6x6 identitäts matrix
    R = np.eye(6)
    
    # Calculate cosine and sine of the angle
    c = np.cos(phi)
    s = np.sin(phi)
    
    # 3x3 Rotationsmatrix definieren:
    r_sub_rotation = [[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]]
    # Rotationsmatrix füllen
    R[0:3, 0:3] = r_sub_rotation
    R[3:6, 3:6] = r_sub_rotation
    
    
    return R

def lokale_steifheitsmatrix_unrotiert(L, A=1, E=1, I=1):
    """
    Parameters
    ----------
    L : Stangenlänge in Metern
    A : Querschnittsfläche der Stange in mm^2
        Default-Wert ist 1.
    E : E-Modul des Stangenmaterials in Newton pro mm^2
        Default-Wert ist 1.
    I : Flächenträgheitsmoment in mm^2 mal m^2
        Default-Wert ist 1.

    Returns
    -------
    R : 6x6 Matrix, welche die Steifheitsmatrix eines einzelnen horizontal orientierten
        Stabes mit linkem Knoten links, rechter Knoten rechts ergibt.

    """
    alpha = A*E/L
    beta = E*I/(L**3)
    
    R=np.zeros((6,6))
    R[0,0] = R[3,3] = alpha
    R[0,3] = R[3,0] = -alpha
    R[1,1] = R[4,4] = 12*beta
    R[2,2] = R[5,5] = 4*L**2 * beta
    R[1,4] = R[4,1] = -12*beta
    R[1,2] = R[2,1] = R[1,5] = R[5,1] = 6*L*beta
    R[2,4] = R[4,2] = R[4,5] = R[5,4] = -6*L*beta
    R[2,5] = R[5,2] = 2*L**2 * beta
    
    return R

def plottePositionen(full_x, stangen_zu_knoten,allphi=None,filename=None, show_labels = True):
    """
    Plottet die Positionen der Stangen.
    
    Parameters
    ----------
    full_x : numpy-array
        Enthält 3* Anzahl der Punkte viele Werte im Schema x1,y1,theta1,x2,y2,theta2,...
    stangen_zu_knoten : dict
        Enthält das Mapping von Stange (==key) zum Tupel (linkerKnoten, rechterKnoten) (==value)
    allphi : numpy-array of angles, optional
        Winkel phi der Stangen in Grad. Default ist None.
    filename : String, optional
        Dateiname zur Speicherung des erstellten plots, Default ist None.
    show_labels : Boolean, optional
        Schalte Label im Plot an oder aus. Default ist True.

    Returns
    -------
    None.

    """
    # Extrahiere die x- und y-Koordinaten aus full_x
    points = [(full_x[i], full_x[i + 1]) for i in range(0, len(full_x), 3)]
    cmap_points = get_cmap('viridis')
    cmap_stangen = get_cmap('plasma')
    
    num_points = len(points)
    points_colors = []
    for point in range(num_points):
        points_colors.append(cmap_points(point/ num_points))

    num_stangen = np.max(list(stangen_zu_knoten.keys()))
    stangen_colors = []
    for stange in range(num_stangen):
        stangen_colors.append(cmap_stangen(stange/ num_stangen))

    # Plotte die Punkte und beschrifte sie
    for i, (x_i, y_i) in enumerate(points):
        plt.plot(x_i, y_i, 'o', label=f'Knoten {i + 1}',color=points_colors[i])
        if show_labels:
            plt.text(x_i, y_i+0.025, f'{i + 1}', fontsize=10, color=points_colors[i],ha='center')
    
    for stange_id, (knoten_links_idx, knoten_rechts_idx) in stangen_zu_knoten.items():
        knoten_links_idx -=1
        knoten_rechts_idx -=1
        stange_id -=1
        x_values = [points[knoten_links_idx][0], points[knoten_rechts_idx][0]]
        y_values = [points[knoten_links_idx][1], points[knoten_rechts_idx][1]]
        
        plt.plot(x_values, y_values, color=stangen_colors[stange_id]) # Verbinde mit blauen Linien


        # Berechne den Mittelpunkt der Verbindungslinie
        mid_point = ((x_values[0] + x_values[1]) / 2,
                      (y_values[0] + y_values[1]) / 2)
        
        # Position für den Winkelindikator etwas horizontal versetzt von Knoten Links
        offset_position = (points[knoten_links_idx][0] + 0.0015,
                           points[knoten_links_idx][1])
        offset_label_x = 0
        offset_label_y = .03
        label_position_x = mid_point[0] + offset_label_x
        label_position_y = mid_point[1] + offset_label_y 
        if show_labels:
            plt.text(label_position_x, label_position_y,
                 fr'$S_{stange_id+1}$', fontsize=10,
                 ha='center', color=stangen_colors[stange_id])
        
        if allphi is not None:
            phi_value = allphi[stange_id]
            if phi_value !=0:
                # Füge einen Teilkreis hinzu
                arc_radius = np.linalg.norm(np.array([x_values[0], y_values[0]]) - np.array([mid_point[0], mid_point[1]])) * 0.5
                
                arc_angle_start = 0   # Startwinkel des Bogens in Grad
                arc_angle_end = phi_value    # Endwinkel des Bogens in Grad
                
                arc = Arc(offset_position,
                          arc_radius * 2,
                          arc_radius * 2,
                          angle=0,
                          theta1=arc_angle_start,
                          theta2=arc_angle_end,
                          color=stangen_colors[stange_id],
                          linewidth=1.5)
                
                plt.gca().add_patch(arc)
                # Berechne die Position für das Label phi_n nahe dem Bogen
                offset_label_x = 0.03
                offset_label_y = 0.0
                label_position_x = offset_position[0] + offset_label_x + arc_radius * np.cos(np.radians((arc_angle_start + arc_angle_end) / 2))
                label_position_y = offset_position[1] + offset_label_y + arc_radius * np.sin(np.radians((arc_angle_start + arc_angle_end) / 2))
                
                if show_labels:
                    plt.text(label_position_x, label_position_y,
                         fr'$\phi_{stange_id+1}$', fontsize=10,
                         ha='center', color=stangen_colors[stange_id])
    
    # Achsen beschriften und Titel hinzufügen
    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    plt.title('Plot der Knoten und Verbindungen')
    plt.grid()
    plt.axis('equal') # Gleiche Skalierung für X und Y-Achse
    if show_labels:
        plt.legend()
    
    if filename is not None:
        plt.savefig(filename, dpi = 300)
    
    plt.show()

def check_Lengths(full_x, stangen_zu_knoten):
    """
    Berechnet alle Stangenlängen des Vectors full_x.

    Parameters
    ----------
    full_x : numpy-array
        Enthält 3* Anzahl der Punkte viele Werte im Schema x1,y1,theta1,x2,y2,theta2,...
    stangen_zu_knoten : dict
        Enthält das Mapping von Stange (==key) zum Tupel (linkerKnoten, rechterKnoten) (==value).

    Returns
    -------
    stangen_längen : list
        Enthält die Längen der einzelnen Stangen berechnet aus den gegebenen Positionen.
    """
    points = [(full_x[i], full_x[i + 1]) for i in range(0, len(full_x), 3)]
    stangen_längen = []
    for stange_id, (knoten_links_idx, knoten_rechts_idx) in stangen_zu_knoten.items():
        knoten_links_idx -=1
        knoten_rechts_idx -=1
        stange_id -=1
        x_values = [points[knoten_links_idx][0], points[knoten_rechts_idx][0]]
        y_values = [points[knoten_links_idx][1], points[knoten_rechts_idx][1]]
        stangen_längen.append(np.sqrt((x_values[1]-x_values[0])**2+(y_values[1]-y_values[0])**2))
    return stangen_längen


def plotMatrix(matrix,filename=None):
    """
    Erstellt einen Matrix-Plot. Die Matrix kann sehr große positive und sehr kleine negative Werte haben.
    Es wird in logarithmischer Skala geplottet, aber das Vorzeichen der Einträge bleibt erhalten.
    0 ist weiß.
    positive Werte sind blau.
    negative Werte sind rot.

    Parameters
    ----------
    matrix : numpy n by n array
        Matrix die geplottet werden soll.
    filename : String, optional
        Dateiname zum Speichern des Plots. Default ist None.

    Returns
    -------
    answer : None
    """
    # Create a masked array where zeros will be set to NaN (to display them as white)
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    to_log_values = np.ma.masked_where(masked_matrix.mask, matrix)
    log_norm_values = np.log10(np.abs(to_log_values)) * np.sign(matrix)
    
    
    # Create a custom colormap: Red for positive values and Blue for negative values
    cmap = plt.get_cmap('RdBu', lut=256)
    max_val = np.max(np.abs(log_norm_values))
    
    # Normalize the color mapping based on logarithmic scale (ignoring zeros)
    norm = Normalize(vmin=-max_val, vmax=max_val)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    cax = plt.imshow(masked_matrix,
                     cmap=cmap,
                     norm=norm,
                     alpha=1) # Overlay colors based on calculated log-normalized values
    
    # Overlay sign-adjusted normalized values with masking
    plt.imshow(log_norm_values, cmap=cmap,
               norm=norm,
               interpolation='nearest')
    
    # Add color bar to indicate value scale
    colorbar = plt.colorbar(cax, label='Value') 

    def scientific_notation(x):
        answer = ""
        if x > 0:
            answer = f'$10^{int(x)}$' 
        elif x<0:
            answer = f'$-10^{int(x)}$' 
        else:
            answer = '0'
        return answer

    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))
    
    # Set ticks and labels for the color bar using logarithmic scale positions
    tick_max = int(np.max(np.abs(log_norm_values)))
    ticks = np.arange(-tick_max,tick_max+1,1) # Define where you want ticks in normalized space.
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([scientific_notation(tick) for tick in ticks])

    
    # Set ticks and labels
    plt.xticks(ticks=np.arange(matrix.shape[1]), labels=np.arange(1, matrix.shape[1]+1))
    plt.yticks(ticks=np.arange(matrix.shape[0]), labels=np.arange(1, matrix.shape[0]+1))
    
    # Label axes
    plt.xlabel('Spalten Index')
    plt.ylabel('Zeilen Index')
    plt.title('Matrixwerte')
    
    if filename is not None:
        plt.savefig(filename,dpi=300)
    
    # Show the plot
    plt.show()