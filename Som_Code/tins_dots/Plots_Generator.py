import io

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import A4

from Som_Code.utils import Utils
from PIL import Image

from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader



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
            print("Trial "+ str(cnt))
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