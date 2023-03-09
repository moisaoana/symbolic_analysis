import io
import matplotlib
import numpy as np
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import A4
from mayavi import mlab

from PIL import Image

from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from Som_Code.utils import Utils

matplotlib.use('Qt5Agg')

class PlotsGenerator:

    @staticmethod
    def generateColorSequenceForTrial(no_trial, trial_data, som):
        colors_sequence = Utils.get_rgb_colors_array(trial_data, som)
        fig = go.Figure()
        # Add a rectangle trace for each segment of the barcode
        for i in range(len(colors_sequence)):
            print('Cs ', i)
            fig.add_shape(
                type='rect',
                x0=i,
                y0=0,
                x1=i + 1,
                y1=3,
                fillcolor=colors_sequence[i],
                line=dict(width=0),
            )

        # Update the layout
        fig.update_layout(
            xaxis=dict(range=[0, len(colors_sequence)], title='time', showticklabels=False),
            yaxis=dict(title='Trial ' + no_trial, range=[0, 3]),
            width=1200,
        )

        return fig, colors_sequence

    @staticmethod
    def generateColorSeguenceForAllTrials(no_trials, all_trials_data, som):

        fig = make_subplots(rows=no_trials, cols=1)

        color_seg_lengths = []

        for cnt, trial in enumerate(all_trials_data):
            color_seq_fig, color_sequence = PlotsGenerator.generateColorSequenceForTrial(cnt, trial, som)
            color_seg_lengths.append(len(color_sequence))
            for shape in color_seq_fig.layout.shapes:
                fig.add_shape(shape, row=cnt + 1, col=1)

        for i in range(no_trials):
            fig.update_xaxes(title_text="Time", range=[0, color_seg_lengths[i]], row=i + 1, col=1)
            fig.update_yaxes(title_text="Trial " + str(i + 1), range=[0, 3], row=i + 1, col=1)

        fig.update_layout(title='Color sequences for each trial')

        fig.show()

    @staticmethod
    def generateColorSeguenceForAllTrialsWithImages(no_trials, all_trials_data, som):
        fig = plt.figure(figsize=(10, 10))
        rows = no_trials
        columns = 1
        for cnt, trial in enumerate(all_trials_data):
            print("Trial " + str(cnt))
            color_seq_fig, color_sequence = PlotsGenerator.generateColorSequenceForTrial(cnt, trial.trial_data, som)
            image_bytes = pio.to_image(color_seq_fig, format='png')
            image = Image.open(io.BytesIO(image_bytes))
            fig.add_subplot(rows, columns, cnt)
            plt.imshow(image)
            plt.axis('off')
            plt.title("Trial " + str(cnt + 1))
        fig.show()

    @staticmethod
    def generateColorSeguenceForAllTrialsInPDF(no_trials, all_trials_data, som):
        PAGE_WIDTH, PAGE_HEIGHT = A4
        c = canvas.Canvas("trials.pdf", pagesize=A4)
        TOP_MARGIN = 0
        BOTTOM_MARGIN = 0
        img_height = 280
        y = PAGE_HEIGHT - TOP_MARGIN - img_height
        for cnt, trial in enumerate(all_trials_data):
            print("Trial " + str(cnt))
            color_seq_fig, color_sequence = PlotsGenerator.generateColorSequenceForTrial(cnt, trial.trial_data, som)
            image_bytes = pio.to_image(color_seq_fig, format='png')
            image = Image.open(io.BytesIO(image_bytes))
            if y >= BOTTOM_MARGIN:
                print("same")
                c.drawImage(ImageReader(image), 10, y, width=600, height=280, preserveAspectRatio=True)
                y -= img_height
            else:
                print("other")
                c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN - img_height
                c.drawImage(ImageReader(image), 10, y, width=600, height=280, preserveAspectRatio=True)
                y -= img_height
        c.save()

    @staticmethod
    def generateColorSequenceForTrialMatplotlib(colors, width=400, height=100):
        color_indices = np.arange(len(colors))
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow([color_indices], cmap=cmap, aspect="auto", extent=(0, width, 0, height))
        ax.set_xticks([])
        ax.set_yticks([])
        print('Barcode generated')
        plt.close()
        return fig, ax, color_indices

    @staticmethod
    def getTrialSequencesArray(all_trials_data, som):
        barcodes_array = []
        all_color_arrays = []
        max_length_color_array = 0
        for cnt, trial in enumerate(all_trials_data):
            print("Trial " + str(cnt))
            colors_array = Utils.get_colors_array(trial.trial_data, som)
            all_color_arrays.append(colors_array)
            if len(colors_array) > max_length_color_array:
                max_length_color_array = len(colors_array)
        print("Max is ",max_length_color_array)
        for cnt, colors_array in enumerate(all_color_arrays):
            if len(colors_array) != max_length_color_array:
                for i in range(0, max_length_color_array-len(colors_array)):
                    colors_array.append([1, 1, 1])
            figure_data_tuple = PlotsGenerator.generateColorSequenceForTrialMatplotlib(colors_array)
            barcodes_array.append(figure_data_tuple)
        return barcodes_array

    # visibilities: 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
    @staticmethod
    def groupByStimulusVisibility(figure_array):
        PlotsGenerator.generateGridWithColorSequences(figure_array[0:30], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.00")
        PlotsGenerator.generateGridWithColorSequences(figure_array[30:60], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.05")
        PlotsGenerator.generateGridWithColorSequences(figure_array[60:90], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.1")
        PlotsGenerator.generateGridWithColorSequences(figure_array[90:120], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.15")
        PlotsGenerator.generateGridWithColorSequences(figure_array[120:150], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.2")
        PlotsGenerator.generateGridWithColorSequences(figure_array[150:180], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.25")
        PlotsGenerator.generateGridWithColorSequences(figure_array[180:210], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.3")
        plt.show()

    @staticmethod
    def groupByResponse(figure_array):
        nothing_figures = [*figure_array[0:61], *figure_array[62:69], *figure_array[70:76], *figure_array[77:90],
                           *figure_array[95:96], *figure_array[104:105], *figure_array[107:108], *figure_array[111:117],
                           *figure_array[120:121]]
        something_figures = [*figure_array[61:62], *figure_array[69:70], *figure_array[76:77], *figure_array[90:95],
                             *figure_array[96:104], *figure_array[105:107], *figure_array[108:111], *figure_array[112:114],
                             *figure_array[118:120], *figure_array[123:124], *figure_array[134:138], *figure_array[139:141],
                             *figure_array[142:143], *figure_array[144:152], *figure_array[156:158], *figure_array[162:163],
                             *figure_array[164:166], *figure_array[169:170], *figure_array[174:175], *figure_array[177:179],
                             *figure_array[180:183]]
        identified_figures = [*figure_array[117:118], *figure_array[121:123], *figure_array[124:132], *figure_array[133:134],
                              *figure_array[138:139], *figure_array[141:142], *figure_array[143:144], *figure_array[152:156],
                              *figure_array[158:162], *figure_array[163:164], *figure_array[166:169], *figure_array[170:174],
                              *figure_array[175:177], *figure_array[179:180], *figure_array[183:210], *figure_array[132:133]]

        PlotsGenerator.generateGridWithColorSequences(nothing_figures, n_rows=len(nothing_figures), n_cols=1)
        plt.suptitle("Response: nothing")
        PlotsGenerator.generateGridWithColorSequences(something_figures, n_rows=len(something_figures), n_cols=1)
        plt.suptitle("Response: something")
        PlotsGenerator.generateGridWithColorSequences(identified_figures, n_rows=len(identified_figures), n_cols=1)
        plt.suptitle("Response: what the subject sees (correct + 1 incorrect)")
        plt.show()

    @staticmethod
    def groupByStimulus(figure_array):

        poseta = [figure_array[0], figure_array[31], figure_array[82],figure_array[106], figure_array[144],
                  figure_array[157], figure_array[185]]
        topor = [figure_array[1], figure_array[49], figure_array[69], figure_array[97], figure_array[143],
                  figure_array[170], figure_array[190]]
        oala = [figure_array[2], figure_array[54], figure_array[85], figure_array[119], figure_array[146],
                  figure_array[159], figure_array[199]]
        elicopter = [figure_array[3], figure_array[41], figure_array[77], figure_array[99], figure_array[133],
                  figure_array[171], figure_array[192]]
        urs = [figure_array[4], figure_array[36], figure_array[86], figure_array[93], figure_array[123],
                  figure_array[169], figure_array[181]]
        palarie = [figure_array[5], figure_array[43], figure_array[83], figure_array[100], figure_array[126],
                  figure_array[154], figure_array[194]]
        foarfece = [figure_array[6], figure_array[59], figure_array[78], figure_array[118], figure_array[142],
                  figure_array[165], figure_array[198]]
        banana = [figure_array[7], figure_array[51], figure_array[70], figure_array[110], figure_array[149],
                  figure_array[155], figure_array[193]]
        lampa = [figure_array[8], figure_array[30], figure_array[74], figure_array[114], figure_array[136],
                  figure_array[156], figure_array[182]]
        chitara = [figure_array[9], figure_array[34], figure_array[76], figure_array[113], figure_array[138],
                  figure_array[166], figure_array[195]]
        masina = [figure_array[10], figure_array[56], figure_array[61], figure_array[102], figure_array[145],
                  figure_array[168], figure_array[203]]
        vaca = [figure_array[11], figure_array[45], figure_array[63], figure_array[94], figure_array[129],
                  figure_array[172], figure_array[206]]
        furculita = [figure_array[12], figure_array[47], figure_array[73], figure_array[90], figure_array[127],
                  figure_array[152], figure_array[205]]
        cerb = [figure_array[13], figure_array[48], figure_array[65], figure_array[117], figure_array[131],
                  figure_array[163], figure_array[189]]
        pantaloni = [figure_array[14], figure_array[35], figure_array[79], figure_array[96], figure_array[134],
                  figure_array[176], figure_array[184]]
        scaun = [figure_array[15], figure_array[52], figure_array[80], figure_array[112], figure_array[130],
                  figure_array[160], figure_array[197]]
        peste = [figure_array[16], figure_array[37], figure_array[89], figure_array[105], figure_array[122],
                  figure_array[161], figure_array[187]]
        caine = [figure_array[17], figure_array[32], figure_array[67], figure_array[91], figure_array[128],
                  figure_array[167], figure_array[191]]
        sticla = [figure_array[18], figure_array[53], figure_array[68], figure_array[95], figure_array[120],
                  figure_array[178], figure_array[204]]
        pistol = [figure_array[19], figure_array[58], figure_array[64], figure_array[101], figure_array[132],
                  figure_array[173], figure_array[200]]
        bicicleta = [figure_array[20], figure_array[39], figure_array[81], figure_array[115], figure_array[140],
                  figure_array[162], figure_array[196]]
        cal = [figure_array[21], figure_array[44], figure_array[75], figure_array[98], figure_array[139],
                  figure_array[179], figure_array[183]]
        elefant = [figure_array[22], figure_array[43], figure_array[88], figure_array[107], figure_array[148],
                  figure_array[174], figure_array[201]]
        iepure = [figure_array[23], figure_array[55], figure_array[60], figure_array[103], figure_array[141],
                  figure_array[151], figure_array[188]]
        pahar = [figure_array[24], figure_array[46], figure_array[72], figure_array[111], figure_array[137],
                  figure_array[164], figure_array[180]]
        masa = [figure_array[25], figure_array[38], figure_array[87], figure_array[109], figure_array[124],
                  figure_array[175], figure_array[207]]
        umbrela = [figure_array[26], figure_array[50], figure_array[66], figure_array[108], figure_array[125],
                  figure_array[150], figure_array[209]]
        fluture = [figure_array[27], figure_array[33], figure_array[62], figure_array[118], figure_array[121],
                  figure_array[153], figure_array[186]]
        girafa = [figure_array[28], figure_array[57], figure_array[71], figure_array[92], figure_array[135],
                  figure_array[158], figure_array[208]]
        pian = [figure_array[29], figure_array[40], figure_array[84], figure_array[104], figure_array[147],
                  figure_array[177], figure_array[202]]

        PlotsGenerator.generateGridWithColorSequences(poseta, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: poseta/geanta (de dama)")
        PlotsGenerator.generateGridWithColorSequences(topor, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: topor/secure")
        PlotsGenerator.generateGridWithColorSequences(oala, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: oala/cratita")
        PlotsGenerator.generateGridWithColorSequences(elicopter, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elicopter")
        PlotsGenerator.generateGridWithColorSequences(urs, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: urs (polar)")
        PlotsGenerator.generateGridWithColorSequences(palarie, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: palarie")
        PlotsGenerator.generateGridWithColorSequences(foarfece, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: foarfece")
        PlotsGenerator.generateGridWithColorSequences(banana, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: banana")
        PlotsGenerator.generateGridWithColorSequences(lampa, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: lampa/veioza")
        PlotsGenerator.generateGridWithColorSequences(chitara, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: chitara (electrica)")
        PlotsGenerator.generateGridWithColorSequences(masina, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masina")
        PlotsGenerator.generateGridWithColorSequences(vaca, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: vaca")
        PlotsGenerator.generateGridWithColorSequences(furculita, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: furculita")
        PlotsGenerator.generateGridWithColorSequences(cerb, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cerb")
        PlotsGenerator.generateGridWithColorSequences(pantaloni, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pantaloni (scurti)")
        PlotsGenerator.generateGridWithColorSequences(scaun, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: scaun")
        PlotsGenerator.generateGridWithColorSequences(peste, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: peste")
        PlotsGenerator.generateGridWithColorSequences(caine, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: caine/catel")
        PlotsGenerator.generateGridWithColorSequences(sticla, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: sticla")
        PlotsGenerator.generateGridWithColorSequences(pistol, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pistol")
        PlotsGenerator.generateGridWithColorSequences(bicicleta, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: bicicleta")
        PlotsGenerator.generateGridWithColorSequences(cal, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cal")
        PlotsGenerator.generateGridWithColorSequences(elefant, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elefant")
        PlotsGenerator.generateGridWithColorSequences(iepure, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: iepure")
        PlotsGenerator.generateGridWithColorSequences(pahar, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pahar/cupa")
        PlotsGenerator.generateGridWithColorSequences(masa, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masa")
        PlotsGenerator.generateGridWithColorSequences(umbrela, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: umbrela")
        PlotsGenerator.generateGridWithColorSequences(fluture, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: fluture")
        PlotsGenerator.generateGridWithColorSequences(girafa, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: girafa")
        PlotsGenerator.generateGridWithColorSequences(pian, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pian")
        plt.show()

    @staticmethod
    def generateGridWithColorSequences(figure_data_array, n_rows=2, n_cols=1, width=400, height=100):
        figure_index = 0
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        for i in range(n_rows * n_cols):
            print('Barcode grid figure index ', figure_index)
            row_idx = i // n_cols
            col_idx = i % n_cols
            _, ax, color_indices = figure_data_array[figure_index]
            figure_index += 1
            axs[row_idx, col_idx].axis('off')
            axs[row_idx, col_idx].imshow(color_indices.reshape(1, -1), cmap=ax.images[0].cmap, aspect='auto')
        plt.subplots_adjust(hspace=0, wspace=0)
        return fig, axs

    @staticmethod
    def generateScatterPlotForDistanceMapMatplotlib(size, distance_map):
        Nx, Ny, Nz = size, size, size
        X, Y, Z = np.arange(Nx), np.arange(Ny), -np.arange(Nz)

        fig = plt.figure(figsize=(size, size))
        ax = Axes3D(fig)

        # Add x, y gridlines
        ax.grid(b=True, color='red',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        kw = {
            'cmap': 'Blues'
        }
        for i in range(0, size):
            for j in range(0, size):
                for k in range(0, size):
                    ax.scatter3D(X[i], Y[j], Z[k], c=str(distance_map[i][j][k]), s=100, cmap=plt.get_cmap('jet'))

        # Show Figure
        plt.show()


    @staticmethod
    def generateScatterPlotForClustersMatplotlib(som, eegDataProcessor, size):
        threshold = som.find_threshold(eegDataProcessor.processed_data)
        print('Max dist ', threshold)
        no_clusters, bmu_array, samples_with_clusters_array = som.find_clusters_with_min_dist(
            eegDataProcessor.processed_data,
            0.3, threshold)
        print('No clusters ', no_clusters)

        fig = plt.figure(figsize=(size, size))
        ax = Axes3D(fig)
        ax.grid(b=True, color='red',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        markers_and_colors = Utils.assign_markers_and_colors(no_clusters)
        for cnt, xx in enumerate(eegDataProcessor.processed_data):
            w = som.find_BMU(xx)
            cluster = 0
            for bmu in bmu_array:
                if w == bmu[0]:
                    cluster = bmu[1]
                    break
            marker = '_'
            color = 'y'
            for x in markers_and_colors:
                if x[0] == cluster:
                    marker = x[1]
                    color = x[2]
            ax.scatter3D(w[0], w[1], w[2], color=color, marker=marker)
        # Show Figure
        plt.show()

    @staticmethod
    def generateScatterPlotForDistanceMapPlotly(size, distance_map):
        colors = []
        X_coord = []
        Y_coord = []
        Z_coord = []

        for i in range(0, size):
            for j in range(0, size):
                for k in range(0, size):
                    X_coord.append(i)
                    Y_coord.append(j)
                    Z_coord.append(k)
                    colors.append(distance_map[i][j][k])

        trace = go.Scatter3d(
            x=X_coord,
            y=Y_coord,
            z=Z_coord,
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.8,
                symbol='circle',
                colorscale='Viridis',
                showscale=True
            ),
            text=colors
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            )
        )

        # Create a figure with the trace and layout, and show the plot
        fig1 = go.Figure(data=[trace], layout=layout)
        fig1.update_layout(
            title='Distance map'
        )
        fig1.show()

    @staticmethod
    def generateScatterPlotForClustersPlotly(som, eegDataProcessor):
        threshold = som.find_threshold(eegDataProcessor.processed_data)
        print('Max dist ', threshold)
        no_clusters, bmu_array, samples_with_clusters_array = som.find_clusters_with_min_dist(
            eegDataProcessor.processed_data,
            0.3, threshold)
        print('No clusters ', no_clusters)
        samples_with_symbols_array = Utils.assign_symbols(samples_with_clusters_array)
        print(samples_with_symbols_array)

        markers_and_colors = Utils.assign_markers_and_colors(no_clusters)
        print(no_clusters)
        print("-------------------------------------")

        BMU_X = []
        BMU_Y = []
        BMU_Z = []
        M = []
        C = []
        BMU = []

        for cnt, xx in enumerate(eegDataProcessor.processed_data):
            w = som.find_BMU(xx)
            # print('W ', w)
            cluster = 0
            for bmu in bmu_array:
                if w == bmu[0]:
                    cluster = bmu[1]
                    break
            marker = '_'
            color = 'y'
            for x in markers_and_colors:
                if x[0] == cluster:
                    marker = x[1]
                    color = x[2]
            notInBMU = True
            for x in BMU:
                if w == x:
                    notInBMU = False
            if notInBMU:
                BMU.append(w)
                BMU_X.append(w[0])
                BMU_Y.append(w[1])
                BMU_Z.append(w[2])
                M.append(marker)
                C.append(color)

        print(M)

        trace = go.Scatter3d(
            x=BMU_X,
            y=BMU_Y,
            z=BMU_Z,
            mode='markers',
            marker=dict(
                size=5,
                color=C,
                opacity=0.8,
                symbol=M
            )
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            )
        )

        # Create a figure with the trace and layout, and show the plot
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(
            title='Clusters given by som'
        )
        fig.show()

    @staticmethod
    def generateSlicerPlotMayavi(distance_map):
        volume_slice_x = mlab.volume_slice(distance_map, plane_orientation='x_axes')
        volume_slice_y = mlab.volume_slice(distance_map, plane_orientation='y_axes')
        volume_slice_z = mlab.volume_slice(distance_map, plane_orientation='z_axes')
        outline = mlab.outline(volume_slice_x)
        colorbar = mlab.colorbar(object=volume_slice_x, title='Data values')
        mlab.show()
